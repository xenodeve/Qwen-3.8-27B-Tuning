# Architecture Decision Records

One decision per file, `NNNN-<kebab>.md`, numbered globally. **A decision is
never edited to reverse it** — write a new ADR and mark the old one
`Superseded by NNNN`, because erasing the history is how the next agent
re-litigates a settled question.

An ADR is warranted when a choice is **hard to reverse**, or when a
**performance or quality decision's rationale would otherwise be lost**. In this
repo the second case is the common one: most decisions here are "we measured
these three and picked one", and the benchmark is the reason.

| # | Decision | Status |
|---|---|---|
| — | none yet | |

**Not yet written, and each is a real candidate:** `q4_0` as the settled KV type
· `ngram-map-k` at 16K and `ngram-mod` at depth · excluding the vendored
llama.cpp build from the repo · three CI checks rather than four. Each was
decided on evidence this session; none has its rationale in one findable place.
