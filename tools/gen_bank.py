#!/usr/bin/env python3
"""wimgpt bank generator.

Pipeline: Wikipedia current-events day pages -> category-filtered event drafts
-> keyword suggestions + guessability flags -> band assignment against
models.json -> draft JSON for human review -> merge reviewed drafts into
questions.json.

The guessability judgment stays human: an answer derivable from pre-event
knowledge (poll leaders, famous sites, ailing leaders) makes a bad item, and
no heuristic catches that reliably. CI opens a draft PR; a person fills
question+truth and drops guessable items before merging.

Usage:
  python3 tools/gen_bank.py --coverage                  # gap counts, no network
  python3 tools/gen_bank.py --check                     # validate bank files
  python3 tools/gen_bank.py --dates 2026-05-01:2026-05-18
  python3 tools/gen_bank.py --since 10                  # last 10 days
  python3 tools/gen_bank.py --thin                      # sample dates in gaps < 3 items
  python3 tools/gen_bank.py --fixture fixture.html --date 2026-05-03   # offline parse test
  python3 tools/gen_bank.py --canaries 3
  python3 tools/gen_bank.py --merge bank-draft.json [--bank questions.json]
"""

from __future__ import annotations

import argparse
import calendar
import json
import random
import re
import sys
import time
import urllib.parse
import urllib.request
from datetime import date, timedelta
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
API = "https://en.wikipedia.org/w/api.php"
UA = "wimgpt-bank-gen/0.1 (maintenance script)"

SKIP_CATEGORIES = {"sports", "ongoing", "recent deaths", "deaths", "obituaries"}
FLAG_PATTERNS = [
    (r"\belect\w*\b|\bvote|\bwin\w*\b|\bdefeat\w*\b|\bballot", "outcome possibly guessable (favorites/polls)"),
    (r"\bdied\b|\bdead\b|\bfuneral\b|\bpass(ed)? away", "death: guessable if ailing"),
    (r"\bfirst\b|\brecord\b|\bunprecedented", "verify uniqueness"),
    (r"\bannounce\w*\b|\breport\w*\b", "rumor/announcement, not settled fact?"),
]
STOPWORDS = set(
    "The A An On In At Of For To By From With As Is Are Was Were His Her Its Their "
    "This That These Those After Before During Under Over Between Both Each More "
    "Most Least New Old Former Later About Following According".split()
)
CALENDAR = set(calendar.month_name[1:] + calendar.day_name[:])

CANARY_TEMPLATES = [
    "{country} announced it would permanently leave the {org}",
    "{country}'s parliament building was damaged in a {hazard}",
    "{company} announced a hostile takeover bid for {company2}",
    "a magnitude {mag} earthquake struck {region}, destroying {n} buildings",
    "{country} banned all {thing} imports overnight",
]
CANARY_POOLS = {
    "org": ["International Labour Organization", "Interpol", "World Customs Organization", "International Telecommunication Union"],
    "hazard": ["fire", "flash flood", "gas explosion"],
    "company": ["Arm Holdings", "Ubisoft", "Grab", "Kioxia", "GoTo"],
    "company2": ["Grammarly", "Figma competitor Canva", "Nuance", "King"],
    "mag": ["5.9", "6.3", "6.8"],
    "region": ["the Azores", "the South Island of New Zealand", "Kamchatka", "the Atlas Mountains"],
    "n": ["300", "700", "1,200"],
    "thing": ["rare earth", "diesel", "beef"],
    "country": ["Portugal", "Chile", "Kenya", "Vietnam", "Belarus", "Peru", "Iceland", "Tunisia",
                "New Zealand", "Norway", "Ghana", "Uruguay", "Laos", "Cyprus", "Bhutan"],
}


# ---------- bank helpers ----------

def load_json(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def gaps(models):
    return sorted({m["cutoff"] for m in models})


def band_of(d, cuts):
    prev = ""
    for c in cuts:
        if (prev == "" or d > prev) and d <= c:
            return f"({prev or 'start'}, {c}]" if prev else f"[start, {c}]"
        prev = c
    return None  # after the newest cutoff: no candidate knows it


def coverage(questions, models):
    cuts = gaps(models)
    real = [q["date"] for q in questions if not q.get("canary")]
    table, empty = [], []
    for a, b in zip([""] + cuts, cuts):
        n = sum(1 for d in real if (a == "" or d > a) and d <= b)
        table.append((a, b, n))
        if n == 0 and (a == "" or (date.fromisoformat(b) - date.fromisoformat(a)).days > 1):
            empty.append((a, b))
    return table, empty


def check(questions, models):
    errs = []
    ids = [q.get("id") for q in questions]
    if len(ids) != len(set(ids)):
        errs.append("duplicate ids")
    canaries = [q for q in questions if q.get("canary")]
    if not canaries:
        errs.append("no canary questions")
    keys = [(bool(q.get("canary")), q.get("date", "")) for q in questions]
    if keys != sorted(keys):
        errs.append("questions not sorted by (canary, date)")
    for q in questions:
        try:
            date.fromisoformat(q.get("date", ""))
        except ValueError:
            errs.append(f"bad date: {q.get('id')}")
        for field in ("id", "question", "truth", "keywords"):
            if field not in q:
                errs.append(f"missing {field}: {q.get('id')}")
    table, empty = coverage(questions, models)
    if empty:
        errs.append(f"empty gaps: {empty}")
    for a, b, n in table:
        print(f"  {a or 'start':>10} -> {b}: {n}")
    if errs:
        print("CHECK FAILED:")
        for e in errs:
            print("  -", e)
        return 1
    print("check: OK")
    return 0


# ---------- wikipedia fetching & parsing ----------

class DayParser(HTMLParser):
    """Collect top-level <li> entries grouped under the nearest h2/h3 heading."""

    def __init__(self):
        super().__init__()
        self.entries = []          # (category, text)
        self.heading = ""
        self._h = None
        self._li_depth = 0
        self._buf = []
        self._capture = False

    def handle_starttag(self, tag, attrs):
        if tag in ("h2", "h3", "h4"):
            self._h = []
            self._capture = True
        elif tag == "li":
            self._li_depth += 1
            if self._li_depth == 1:
                self._buf = []
                self._capture = True
        elif tag in ("ul", "div") and self._li_depth == 0:
            pass  # structure tolerance: rely on li/h only

    def handle_endtag(self, tag):
        if tag in ("h2", "h3", "h4") and self._h is not None:
            self.heading = " ".join("".join(self._h).split())
            self._h, self._capture = None, (self._li_depth > 0)
        elif tag == "li":
            if self._li_depth == 1 and self._buf is not None:
                text = re.sub(r"\[\d+\]", "", " ".join("".join(self._buf).split()))
                if text:
                    self.entries.append((self.heading, text))
            self._li_depth -= 1
            self._capture = self._li_depth > 0
            self._buf = [] if self._li_depth == 0 else self._buf

    def handle_data(self, data):
        if self._capture and (self._h is not None or self._li_depth > 0):
            if self._h is not None:
                self._h.append(data)
            else:
                self._buf.append(data)


def fetch_day(d: date) -> str:
    title = f"Portal:Current events/{d.year} {calendar.month_name[d.month]} {d.day}"
    qs = urllib.parse.urlencode({"action": "parse", "page": title, "format": "json",
                                 "prop": "text", "disablelimitreport": 1})
    req = urllib.request.Request(f"{API}?{qs}", headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode())
    return data["parse"]["text"]["*"]


def parse_day(html):
    p = DayParser()
    p.feed(html)
    return p.entries


# ---------- draft generation ----------

def flags_for(text):
    return [note for pat, note in FLAG_PATTERNS if re.search(pat, text, re.I)]


def keywords_draft(text):
    out = []
    for m in re.findall(r"\b[A-Z][A-Za-z'’\-]+(?:\s+[A-Z][A-Za-z'’\-]+)*\b", text):
        words = m.split()
        if all(w in STOPWORDS or w in CALENDAR for w in words):
            continue
        out.append(m)
    out += re.findall(r"\b\d[\d,\.]*\b", text)
    seen, uniq = set(), []
    for k in out:
        if k not in seen:
            seen.add(k)
            uniq.append(k)
    return uniq[:10]


def token_set(s):
    return {t.lower() for t in re.findall(r"[a-z]{3,}", s)}


def draft_entries(day_entries, d, models, questions):
    cuts = gaps(models)
    bank_text = " ".join(q.get("question", "") + " " + q.get("truth", "") for q in questions)
    bank_tokens = token_set(bank_text)
    drafts, skipped = [], {"category": 0, "date": 0, "dupe": 0}
    for i, (cat, text) in enumerate(day_entries, 1):
        if any(s in cat.lower() for s in SKIP_CATEGORIES):
            skipped["category"] += 1
            continue
        band = band_of(d.isoformat(), cuts)
        if band is None:
            skipped["date"] += 1  # after newest cutoff: nobody knows it
            continue
        if len(token_set(text) & bank_tokens) > 0.45 * max(1, len(token_set(text))):
            skipped["dupe"] += 1
            continue
        drafts.append({
            "id": f"d-{d.isoformat()}-{i:02d}",
            "date": d.isoformat(),
            "band": band,
            "category": cat,
            "source": text,
            "keywords_draft": keywords_draft(text),
            "flags": flags_for(text),
            "question": "",
            "truth": "",
        })
    return drafts, skipped


# ---------- canaries ----------

def gen_canaries(n, questions, rng):
    bank_tokens = token_set(" ".join(q.get("question", "") + " " + q.get("truth", "") for q in questions))
    real_dates = [q["date"] for q in questions if not q.get("canary")]
    lo = date.fromisoformat(min(real_dates))
    hi = date.fromisoformat(max(real_dates))
    out = []
    tries = 0
    while len(out) < n and tries < n * 40:
        tries += 1
        tpl = rng.choice(CANARY_TEMPLATES)
        args = {k: rng.choice(v) for k, v in CANARY_POOLS.items()}
        while "company2" in tpl and args.get("company") == args.get("company2"):
            args["company2"] = rng.choice(CANARY_POOLS["company2"])
        sent = tpl.format(**args)
        if len(token_set(sent) & bank_tokens) > 0.4 * len(token_set(sent)):
            continue
        d = lo + timedelta(days=rng.randrange((hi - lo).days + 1))
        out.append({
            "id": f"canary-{d.isoformat()}-{len(out)+1:02d}",
            "date": d.isoformat(),
            "band": "canary",
            "category": "fabricated",
            "source": sent,
            "keywords_draft": [],
            "flags": [],
            "canary": True,
            "question": f"On {d.strftime('%B %-d, %Y')}, {sent}",
            "truth": "Fictional: no such event on that date.",
        })
    return out


# ---------- merge ----------

def merge(draft_path, bank_path):
    draft = load_json(draft_path)
    bank = load_json(bank_path)
    added = 0
    for e in draft.get("drafts", []) + draft.get("canaries", []):
        if not e.get("question") or not e.get("truth"):
            continue
        if any(q["id"] == e["id"] for q in bank["questions"]):
            continue
        item = {
            "id": e["id"], "date": e["date"], "question": e["question"],
            "truth": e["truth"], "keywords": e.get("keywords", []),
        }
        if e.get("canary"):
            item["canary"] = True
        bank["questions"].append(item)
        added += 1
    bank["questions"].sort(key=lambda q: (bool(q.get("canary")), q["date"]))
    with open(bank_path, "w", encoding="utf-8") as f:
        json.dump(bank, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"merged {added} items into {bank_path}")
    return 0


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dates", help="A:B inclusive, ISO")
    ap.add_argument("--since", type=int, metavar="N", help="last N days")
    ap.add_argument("--thin", action="store_true", help="sample dates in gaps with < THIN items")
    ap.add_argument("--thin-min", type=int, default=3)
    ap.add_argument("--max-fetch", type=int, default=40)
    ap.add_argument("--fixture", help="parse this HTML file instead of fetching (offline test)")
    ap.add_argument("--date", help="day for --fixture, ISO")
    ap.add_argument("--canaries", type=int, metavar="N")
    ap.add_argument("--with-canaries", type=int, metavar="N",
                    help="also generate N canaries into the same draft file")
    ap.add_argument("--merge", metavar="DRAFT.json")
    ap.add_argument("--bank", default=str(ROOT / "questions.json"))
    ap.add_argument("--models", default=str(ROOT / "models.json"))
    ap.add_argument("-o", "--out", default="bank-draft.json")
    ap.add_argument("--coverage", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    questions = load_json(args.bank)["questions"]
    models = load_json(args.models)["models"]

    if args.coverage or not any([args.dates, args.since, args.thin, args.fixture,
                                 args.canaries, args.merge, args.check]):
        table, empty = coverage(questions, models)
        for a, b, n in table:
            print(f"  {a or 'start':>10} -> {b}: {n}")
        if empty:
            print(f"thin/empty gaps: {empty}")
        return 0
    if args.check:
        return check(questions, models)
    if args.merge:
        return merge(args.merge, args.bank)
    if args.canaries:
        rng = random.Random(args.seed)
        cans = gen_canaries(args.canaries, questions, rng)
        payload = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S"), "drafts": [], "canaries": cans}
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        for c in cans:
            print(f"  canary {c['date']}: {c['question']}")
        print(f"DRAFTS: 0\nCANARIES: {len(cans)} -> {args.out}")
        return 0

    # date list
    if args.fixture:
        days = [date.fromisoformat(args.date or "2026-05-03")]
    elif args.dates:
        a, b = args.dates.split(":")
        days = [date.fromisoformat(a) + timedelta(days=i)
                for i in range((date.fromisoformat(b) - date.fromisoformat(a)).days + 1)]
    elif args.since:
        today = date.today()
        days = [today - timedelta(days=i) for i in range(args.since, -1, -1)]
    elif args.thin:
        table, _ = coverage(questions, models)
        targets = [(a, b) for a, b, n in table if n < args.thin_min and not (a and (date.fromisoformat(b) - date.fromisoformat(a)).days <= 1)]
        days = []
        for a, b in targets:
            lo = date.fromisoformat(a) if a else min(date.fromisoformat(q["date"]) for q in questions if not q.get("canary"))
            hi = date.fromisoformat(b)
            span = max(1, (hi - lo).days)
            step = max(1, span // max(1, args.max_fetch // max(1, len(targets))))
            days += [lo + timedelta(days=i) for i in range(0, span + 1, step)]
        days = sorted(set(days))[: args.max_fetch]
    else:
        ap.error("nothing to do")

    all_drafts, total_skipped = [], {"category": 0, "date": 0, "dupe": 0}
    for d in days:
        try:
            html = Path(args.fixture).read_text(encoding="utf-8") if args.fixture else fetch_day(d)
        except Exception as e:  # noqa: BLE001
            print(f"  {d}: fetch failed: {e}", file=sys.stderr)
            continue
        ds, skipped = draft_entries(parse_day(html), d, models, questions)
        all_drafts += ds
        for k in total_skipped:
            total_skipped[k] += skipped[k]
        print(f"  {d}: {len(ds)} drafts (skipped {skipped})")

    payload = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "skipped": total_skipped,
        "drafts": all_drafts,
        "canaries": [],
        "instructions": "Fill question+truth for kept items (keywords = grading fallback), "
                        "delete guessable ones, optionally add canaries, then: "
                        "python3 tools/gen_bank.py --merge bank-draft.json",
    }
    if args.with_canaries:
        rng = random.Random(args.seed)
        payload["canaries"] = gen_canaries(args.with_canaries, questions, rng)
        print(f"CANARIES: {len(payload['canaries'])}")
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"DRAFTS: {len(all_drafts)} -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
