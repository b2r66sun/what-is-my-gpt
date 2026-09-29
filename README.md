# wimgpt — which model are you talking to?

Estimates which GPT model is actually serving you, by probing its training-data
cutoff: send a quiz about dated real events, paste the reply back, and get a
posterior over candidate models plus a downgrade probability. Static page, no
build step.

## Use

1. Open `index.html` over http — locally `python3 -m http.server`, or push the
   repo to GitHub and enable Pages on the root.
2. Copy the quiz, send it in a fresh ChatGPT chat with search disabled.
3. Paste the reply, hit Analyze. (Load sample → Analyze for a demo.)

## How it works

Quiz of `n` dated events `q_1..q_n` (event date `d_i`), candidates `m_1..m_M`
with training cutoff `c_j` and prior `π_j`. Reply `r_i ∈ {0,1}`: knows the event.

Likelihood — answers conditionally independent given the model:

```
P(r_i = 1 | m_j) = p      if d_i <= c_j     (in window, p ≈ 0.9)
                   eps    if d_i >  c_j     (past cutoff: guess/hallucination)
```

Both parameters deviate from the naive "0/1" on purpose: `eps > 0` so a single
lucky guess or a predictable event can't zero out a hypothesis, `p < 1` because
in-window events can still fail recall. With `n_j, K_j, G_j` = questions before
cutoff / correct before / "correct" after, the likelihood collapses to

```
L_j = p^K_j (1-p)^(n_j-K_j) · eps^G_j (1-eps)^(n-n_j-G_j)
```

Posterior is a softmax of `log π_j + log L_j`. Output: per-model posterior,
MAP, downgrade probability (sum over `trash` profiles), posterior entropy, and
a canary check (fabricated events — claiming to know one flags hallucination).

A single boundary question is worth `log(p/eps) ≈ 3.4` nats if answered
correctly and `log((1-eps)/(1-p)) ≈ 2.3` nats against if not — so put 2–3
questions in each gap between candidate cutoffs.

The model is asked to reply in JSON (`knows` 0/1 self-report); if it doesn't,
numbered lines are parsed and graded by keyword match; missing = 0.

## Data files

`questions.json` — dated events. Rules: place events in the gaps between
candidate cutoffs; avoid predictable events (eclipses, scheduled election days)
— anything guessable needs a high `eps`; lower `p` for obscure events; keep the
two canaries fabricated and rotate them periodically; a public quiz eventually
enters training data, so keep your bank private.

`models.json` — candidate profiles. Cutoffs as of 2026-09 (partly third-party
reporting; verify against developers.openai.com):

| model | cutoff |
|---|---|
| gpt-6-astra | 2026-04-30 |
| gpt-6-sol | 2026-04-20 |
| gpt-5.6 | 2026-02-28 |
| gpt-5.5-pro / 5.5-mini | 2025-12-01 |
| gpt-5.4 / 5.2 | 2025-08-31 |
| gpt-5 / 5.1 | 2024-09-30 |
| gpt-5-mini / nano | 2024-05-31 |
| gpt-4.1 / o3 | 2024-06-01 |
| gpt-4o / 4o-mini | 2023-10-01 |

## Limitations

- Measures the cutoff, not the model: candidates sharing a cutoff are only
  separated by priors (e.g. gpt-5.5-pro vs 5.5-mini caps that band's downgrade
  probability near 66% under the default priors — an honest floor, not a bug).
- Web search destroys the test (all-knowing pattern collapses onto the newest
  cutoff). Use a fresh chat with search off; canary hits are the strongest
  "it's making things up" signal.
- Self-reported `knows` can be overconfident; canary hits quantify that.
- Asking all questions in one conversation slightly leaks context between
  items.
- Refusals depress `p` (biased toward older models); keep phrasing neutral.
- Cutoffs change silently; re-verify the table occasionally.

## Sources

- Cutoffs: [llm-knowledge-cutoff-dates](https://github.com/HaoooWang/llm-knowledge-cutoff-dates),
  [GPT-6 Astra launch](https://openai.com/index/gpt-6-astra/),
  [GPT-5.5 Pro cutoff](https://tutorsbot.com), [GPT-5.6 cutoff](https://www.eesel.ai),
  [GPT-6 Astra cutoff](https://dev.to)
- Event dates: Wikipedia current-events portals
  [2025-10](https://en.wikipedia.org/wiki/Portal:Current_events/October_2025),
  [2026-02](https://en.wikipedia.org/wiki/Portal:Current_events/February_2026),
  [2026-03](https://en.wikipedia.org/wiki/Portal:Current_events/March_2026)
- Downgrade scenario: [community report: 5.6 Pro auto-routed to 5.5 mini](https://community.openai.com)
