# plans — what we intend to run, and what we asked outsiders

**These are intent, not results.** For results see
[`../reports/`](../reports/).

| # | document | what it is | status |
|---|---|---|---|
| 00 | [Optimization plan](00-OPTIMIZATION-PLAN.md) | the original flag-tuning plan for Q4 | executed, superseded |
| 01 | [V3 Q1/Q2 test plan](01-V3-Q1-Q2-TEST-PLAN.md) | staged plan after Unsloth republished the repo | executed; reviewed by three agents in [report 14](../reports/14-PANEL-REVIEW.md) |
| 02 | [Research brief](02-RESEARCH-BRIEF-OPTIMIZATION-SURFACE.md) | self-contained brief sent to an external researcher, with hardware, prior results, and acceptance criteria. §10 records the final dispatched plan verbatim | sent; two replies reviewed in [17](../reports/17-EXTERNAL-RESEARCH-REVIEW.md), [18](../reports/18-RESEARCH-ROUND2-REVIEW.md) |
| 03 | [16-layer programme](03-SIXTEEN-LAYER-PROGRAMME.md) | every tunable layer, ordered by expected value on tok/s, context, VRAM and quality | executing; results in [report 20](../reports/20-SIXTEEN-LAYER-RESULTS.md) |

---

## Why 02 is worth reading even though it went to an outsider

It is the only document that states the machine, the workload, the metric, the
prior results and the measurement rules **in one place for someone with no
access to this machine**. If you need to explain this project to anything
outside it, start from that.

Its §4.3 lists the four ways the *previous* research went wrong, and §10 lists
the acceptance criteria every claim has to pass. Both exist because a reply that
looks thorough and invents its numbers costs more time than no reply at all.
