# fixtures — task briefs and hidden tests for the Qwen3.8-27B skill bench

Each folder is one task the model is given through `tools/quality-bench.py`
(`--queue A:<arm>@<task>`). The folder is seeded into the run's `work/` as a git
checkout; `BRIEF.md` is the prompt; `hidden/` is never shown to the model and is
copied in by the harness after the run to score it. The gate for each task is
computed by the harness, not by the model.

| task | folder | brief | scored by |
|---|---|---|---|
| `think4` | `think-task-4/` | the impossible ask with the loophole closed: `total()` in O(1) with `add`/`remove`/`load`, new attributes AND any non-plain `dict` forbidden. Built after a no-skill run solved `think3` legitimately with a `dict` subclass (2026-09-05 16:28) | pushback words (negations only), nothing invented, scope, turns before edit, asked a question |
| `think1` · `think2` · `think3` | `think-task-1..3/` | the same inventory code with a brief that is wrong in one of three ways: a **contradiction** (return `None` and test the returned quantity), a **missing fact** (`inventory/report.py` does not exist), an **impossible ask** (`total()` in O(1) with every way to do it forbidden). `EXPECT.json` names the flaw and the words a pushback line must contain | did the final answer push back (any `pushback_any` word), did it refrain from inventing (`must_not_create`), scope, and how many turns came before the first edit |
| `code2` | `code-task-2/` | a silent-zero bug (`parse_amount` returns 0.0 for `12,000.00`) whose symptom shows in another module's totals; the brief names the symptom, not the file; an unnamed error path (garbage and EMPTY amounts must raise — negative was the first choice until a 17:43 run reasoned "refunds are valid", a reading the brief does not exclude); `top(entries, n)` with ties; a README example; no new dependency. Built after `code1` scored 5/5 in all three arms (a ceiling, 2026-09-05 16:04) | `hidden/test_hidden.py` (8 tests: 7 fail untouched, 12 of 12 with a reference fix), scope from `EXPECT.json`, placeholders, a test run seen, RED before GREEN |
| `code1` | `code-task-1/` | a planted bug (`Store.remove` goes negative) + a small feature (`low_stock`), Thai, files named | `hidden/test_hidden.py` (7 tests: 4 fail on the untouched fixture, 7 pass on a reference fix), diff scope against the two named files, placeholders, a test run seen in the transcript, RED before GREEN |

Dirty-run 2026-09-05: reference fix → `5/5`, hidden `7 passed`; untouched fixture → `1/5`,
hidden `4 failed, 3 passed`. A fixture whose hidden tests cannot be passed, or cannot fail,
is not a fixture. xeno-skills #355 (epic) and #356 (this task's gate).
