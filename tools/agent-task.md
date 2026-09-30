# Agent task: bank drafts from current-events context

You are maintaining the question bank of `wimgpt` — a tool that identifies
which LLM is serving a user by probing training-data cutoffs. Input is
`raw-context.md` (flattened Wikipedia current-events pages; links annotated as
`(->Article title)`, plus an "Article summaries" section). Turn it into
`bank-draft.json`.

Also read `questions.json` first: events already covered by existing questions
must be skipped (the bank may already contain them under different wording).

## Rules

1. **Extract events** from the dated sections. Ignore categories that make bad
   quiz items by nature (sports results, ongoing conflicts without a dated
   development, deaths of people whose death was foreseeable).
2. **Verify dates.** Cross-check each event's section date against the
   "Article summaries" (and the linked article title). If the summary
   contradicts the date, use the correct date and say so in `date_check`.
   If you cannot verify, `date_check: "unverified"` — do not guess.
3. **Judge guessability.** The answer must not be derivable from knowledge
   predating the event: poll leaders who won, ailing leaders who died, famous
   sites that were obviously hit, championship favorites, anything scheduled
   or predictable. Drop such items entirely. When in doubt, drop.
4. **Phrase the question.** One sentence, English, contains the date, must
   not contain or strongly hint at the answer. Prefer asking for a
   distinctive specific (name, number, codename) over "what happened".
5. **Truth**: one short sentence with the distinctive answer.
6. **Keywords** (grading fallback, 1–4): distinctive substrings of plausible
   correct answers, normalized (lowercase, no punctuation); include a CJK
   variant when the term has a common Chinese rendering.

## Output contract

Write `bank-draft.json` — strict JSON, nothing else:

```json
{
  "drafts": [
    {"id": "e-2026-05-03-04",
     "date": "2026-05-03",
     "question": "In early May 2026, which US airline ceased operations after a failed bailout?",
     "truth": "Spirit Airlines.",
     "keywords": ["spirit"],
     "date_check": "verified | corrected to YYYY-MM-DD | unverified"}
  ],
  "canaries": []
}
```

Keep dates ISO; assign ids as `e-<date>-<nn>`; 0–6 items per day; never invent
events; pass through canaries from `raw-context.canaries.json` unchanged if
that file exists.

Then a human (or you, in the same run) merges:
`python3 tools/gen_bank.py --merge bank-draft.json`
