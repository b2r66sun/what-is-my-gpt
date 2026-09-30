#!/usr/bin/env python3
"""wimgpt bank generator (deterministic layer).

Fetches Wikipedia current-events day pages and flattens them (tag-level, no
structural parsing) into raw-context.md, marking each day with the cutoff gap
it belongs to (or RESERVE for dates after the newest cutoff), plus article
summaries (newest-first budget, retried) for date cross-checks. Entry
extraction, question phrasing, keyword choice and guessability judgment are
the agent's job (tools/agent-task.md).

Usage:
  python3 tools/gen_bank.py --coverage                  # gap counts, no network
  python3 tools/gen_bank.py --check                     # validate bank + models
  python3 tools/gen_bank.py --context --since 14        # last 14 days -> raw-context.md
  python3 tools/gen_bank.py --context --dates 2026-05-01:2026-05-18
  python3 tools/gen_bank.py --context --thin            # seeded sampling in gaps < 3 items
  python3 tools/gen_bank.py --context --fixture tools/fixture.html --date 2026-05-03
  python3 tools/gen_bank.py --canaries 3                # -> canaries.json
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
REST = "https://en.wikipedia.org/api/rest_v1/page/summary/"
UA = "wimgpt-bank-gen/0.1 (maintenance script)"

LINK_NS_SKIP = ("Category:", "Portal:", "Special:", "Wikipedia:", "File:", "Help:", "Template:")
# Generic geography pages carry no date-verification value; don't spend the
# summary budget on them.
GENERIC_TITLES = {
    "Ukraine", "Russia", "Kyiv", "Moscow", "Poland", "Germany", "France", "Italy",
    "Spain", "United Kingdom", "United States", "China", "Japan", "India", "Israel",
    "Iran", "Gaza Strip", "West Bank", "Lebanon", "Syria", "Turkey", "Egypt", "Iraq",
    "Saudi Arabia", "Brazil", "Mexico", "Argentina", "Canada", "Australia",
    "South Korea", "North Korea", "Taiwan", "European Union", "NATO", "United Nations",
    "Africa", "Europe", "Asia", "Middle East", "London", "Paris", "Berlin", "Tokyo",
    "Beijing", "Washington, D.C.", "New York City", "Brussels", "Gaza City",
}

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
    "company2": ["Grammarly", "Canva", "Nuance", "King"],
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


def gap_label(d, cuts):
    """(label, note) for a date against candidate cutoffs."""
    prev = ""
    for c in cuts:
        if (prev == "" or d > prev) and d <= c:
            return (f"GAP ({prev or 'start'}, {c}]",
                    f"events here are known by candidates with cutoff >= {c} and unknown below — "
                    f"they discriminate the {c} band")
        prev = c
    return ("RESERVE",
            f"after the newest cutoff ({cuts[-1]}): NO current candidate knows these — "
            f"do NOT draft items from this section")


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
        errs.append("duplicate question ids")
    if not models:
        errs.append("models.json empty")
    mids = [m.get("id") for m in models]
    if len(mids) != len(set(mids)):
        errs.append("duplicate model ids")
    for m in models:
        try:
            date.fromisoformat(m.get("cutoff", ""))
        except ValueError:
            errs.append(f"bad model cutoff: {m.get('id')}")
    if not [q for q in questions if q.get("canary")]:
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
        if not q.get("canary") and not q.get("keywords"):
            errs.append(f"empty keywords on non-canary: {q.get('id')}")
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


# ---------- wikipedia fetching & flattening ----------
#
# No structural parsing on purpose: entry extraction, heading grouping and
# link association are the agent's job. We only flatten tags (headings to
# "## ", list items to "- ", wiki links to "(->Title)" — the first link of
# each bullet is marked "(=>Title)" as the likely subject) — a tag-level
# transform with no assumptions about nesting that survives markup changes.

class Flattener(HTMLParser):
    def __init__(self):
        super().__init__()
        self.out = []
        self.skip = 0
        self.link = None   # (title, is_subject) for the open <a>
        self.fresh = True  # no link emitted since the current bullet/heading began

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "sup"):
            self.skip += 1
        elif tag in ("h1", "h2", "h3", "h4", "li"):
            if tag == "li":
                self.out.append("\n- ")
            else:
                self.out.append("\n\n## ")
            self.fresh = True
        elif tag in ("p", "div", "ul", "table", "tr"):
            self.out.append("\n")
        elif tag == "br":
            self.out.append("\n")
        elif tag == "a":
            href = dict(attrs).get("href", "")
            if href.startswith("/wiki/"):
                title = urllib.parse.unquote(href[6:].split("#")[0]).replace("_", " ")
                if not title.startswith(LINK_NS_SKIP):
                    self.link = (title, self.fresh)

    def handle_endtag(self, tag):
        if tag in ("script", "style", "sup"):
            self.skip = max(0, self.skip - 1)
        elif tag == "a" and self.link is not None:
            title, subject = self.link
            self.out.append((" (=>" if subject else " (->") + title + ")")
            self.link = None
            self.fresh = False
        elif tag in ("h1", "h2", "h3", "h4"):
            self.out.append("\n")

    def handle_data(self, data):
        if not self.skip:
            self.out.append(data if data.strip() else " ")


def flatten(html):
    f = Flattener()
    f.feed(html)
    text = "".join(f.out)
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def day_titles(flat):
    """Subject links first (=>), then other links; multi-word before single-word."""
    subjects = re.findall(r"\(=>([^)]+)\)", flat)
    others = re.findall(r"\(->([^)]+)\)", flat)
    seen, ordered = set(), []
    for group in (subjects, others):
        group.sort(key=lambda t: -len(t.split()))
        for t in group:
            if t not in seen and t not in GENERIC_TITLES:
                seen.add(t)
                ordered.append(t)
    return ordered


def allocate_summaries(day_titles_list, max_sources):
    """Round-robin across days, newest first, so old days can't eat the budget."""
    queues = [list(ts) for ts in day_titles_list]  # caller passes newest-first
    picked, out = set(), []
    while queues and len(out) < max_sources:
        progressed = False
        for q in queues[:]:
            while q:
                t = q.pop(0)
                if t not in picked:
                    picked.add(t)
                    out.append(t)
                    progressed = True
                    break
            if not q:
                queues.remove(q)
        if not progressed:
            break
    return out


def fetch_day(d: date) -> str:
    title = f"Portal:Current events/{d.year} {calendar.month_name[d.month]} {d.day}"
    qs = urllib.parse.urlencode({"action": "parse", "page": title, "format": "json",
                                 "prop": "text", "disablelimitreport": 1})
    req = urllib.request.Request(f"{API}?{qs}", headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode())
    return data["parse"]["text"]["*"]


def fetch_article_summary(title: str, retries: int = 2):
    err = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(
                REST + urllib.parse.quote(title.replace(" ", "_"), safe=""),
                headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode())
            return {"title": data.get("title"), "extract": data.get("extract", "")[:600]}
        except Exception as e:  # noqa: BLE001
            err = repr(e)
            time.sleep(0.5 * (attempt + 1))
    return {"title": title, "extract": "", "error": err}


# ---------- canaries ----------

def token_set(s):
    return {t.lower() for t in re.findall(r"[a-z]{3,}", s)}


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
            "question": f"On {d.strftime('%B %-d, %Y')}, {sent}",
            "truth": "Fictional: no such event on that date.",
            "keywords": [],
            "canary": True,
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
    ap.add_argument("--context", action="store_true",
                    help="fetch day pages, emit flattened context (required for --dates/--since/--thin/--fixture)")
    ap.add_argument("--dates", help="A:B inclusive, ISO")
    ap.add_argument("--since", type=int, metavar="N", help="last N days")
    ap.add_argument("--thin", action="store_true", help="seeded sampling in gaps with < THIN items")
    ap.add_argument("--thin-min", type=int, default=3)
    ap.add_argument("--max-fetch", type=int, default=40)
    ap.add_argument("--max-sources", type=int, default=25, help="article-summary budget per run")
    ap.add_argument("--fixture", help="parse this HTML file instead of fetching (offline test)")
    ap.add_argument("--date", help="day for --fixture, ISO")
    ap.add_argument("--canaries", type=int, metavar="N")
    ap.add_argument("--with-canaries", type=int, metavar="N", help="also generate N canaries")
    ap.add_argument("--merge", metavar="DRAFT.json")
    ap.add_argument("--bank", default=str(ROOT / "questions.json"))
    ap.add_argument("--models", default=str(ROOT / "models.json"))
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--coverage", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    if args.out is None:
        args.out = "canaries.json" if args.canaries and not args.context else "raw-context.md"

    questions = load_json(args.bank)["questions"]
    models = load_json(args.models)["models"]

    if args.coverage or not any([args.context, args.canaries, args.merge, args.check]):
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
        print(f"CANARIES: {len(cans)} -> {args.out}")
        return 0

    if not args.context:
        ap.error("--dates/--since/--thin/--fixture require --context")
    if not any([args.dates, args.since, args.thin, args.fixture]):
        ap.error("--context needs --dates / --since / --thin / --fixture")

    cuts = gaps(models)

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
    else:  # --thin: seeded sampling inside under-filled gaps
        rng = random.Random(args.seed)
        table, _ = coverage(questions, models)
        targets = [(a, b) for a, b, n in table
                   if n < args.thin_min
                   and not (a and (date.fromisoformat(b) - date.fromisoformat(a)).days <= 1)]
        per_gap = max(3, args.max_fetch // max(1, len(targets)))
        days = []
        for a, b in targets:
            lo = date.fromisoformat(a) if a else min(date.fromisoformat(q["date"])
                                                     for q in questions if not q.get("canary"))
            hi = date.fromisoformat(b)
            span = [lo + timedelta(days=i) for i in range((hi - lo).days + 1)]
            days += rng.sample(span, min(len(span), per_gap))
        days = sorted(set(days))[: args.max_fetch]

    sections, per_day_titles, fetched = [], [], 0
    for d in days:  # ascending; summaries are allocated newest-first below
        try:
            html = Path(args.fixture).read_text(encoding="utf-8") if args.fixture else fetch_day(d)
        except Exception as e:  # noqa: BLE001
            print(f"  {d}: fetch failed: {e}", file=sys.stderr)
            continue
        fetched += 1
        flat = flatten(html)
        label, note = gap_label(d.isoformat(), cuts)
        sections.append(f"# {d.isoformat()} — {label}\n({note})\n\n{flat}")
        per_day_titles.append((d, day_titles(flat)))
        print(f"  {d}: flattened [{label}], {len(per_day_titles[-1][1])} candidate sources")

    if fetched == 0:
        print("CONTEXT: 0 days — all fetches failed", file=sys.stderr)
        return 1

    # summary budget: newest days first, round-robin across days
    ordered = allocate_summaries([ts for _, ts in reversed(per_day_titles)], args.max_sources)
    src_parts, failures = [], 0
    for t in ordered:
        if args.fixture:
            src_parts.append(f"## {t}\n(summary not fetched: fixture mode)\n")
            continue
        s = fetch_article_summary(t)
        if s.get("error"):
            failures += 1
            src_parts.append(f"## {t}\n(FETCH FAILED: {s['error']})\n")
        else:
            src_parts.append(f"## {t}\n{s.get('extract') or '(no extract)'}\n")
        time.sleep(0.2)

    cov_lines = [f"  {a or 'start':>10} -> {b}: {n}" for a, b, n in coverage(questions, models)[0]]
    doc = (
        "<!-- wimgpt bank context: feed to the agent task in tools/agent-task.md -->\n"
        "# Current-events context\n\n"
        "# Coverage (drafting priority: thinnest gaps first; never draft from RESERVE)\n"
        + "\n".join(cov_lines) + "\n\n"
        + "\n\n".join(sections) +
        "\n\n# Article summaries (for date cross-checks)\n\n" + "\n".join(src_parts)
    )
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(doc)
    print(f"CONTEXT: {fetched} days -> {args.out}")
    print(f"SUMMARIES: {len(ordered)} (failures: {failures})")
    if args.with_canaries:
        rng = random.Random(args.seed)
        cans = gen_canaries(args.with_canaries, questions, rng)
        cf = Path(args.out).with_suffix(".canaries.json")
        with open(cf, "w", encoding="utf-8") as f:
            json.dump({"canaries": cans}, f, ensure_ascii=False, indent=2)
        print(f"CANARIES: {len(cans)} -> {cf}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
