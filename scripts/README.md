# scripts — tools for the documentation map itself

**Not the measurement scripts.** Those live in
[`../qwen38-tuning/scripts/`](../qwen38-tuning/scripts/README.md) — launch
profiles, `swap-model.sh`, and the unattended `afk-*.sh` queues. Nothing in this
folder touches the GPU or port 8080.

| file | what it does |
|---|---|
| `check-doc-links.py` | every relative markdown link under `C:\AI` must resolve to a real file |
| `audit-stale-claims.py` | finds lines that still state a claim the project later contradicted |

## `check-doc-links.py`

```powershell
python C:\AI\scripts\check-doc-links.py
```

Exit 0 when every link resolves; 1 with the broken ones listed.

The docs are a navigation map — a `README.md` in every folder pointing at the
next one — so a dead link is worse than no link: an agent that follows it
concludes the target does not exist rather than that the pointer is wrong.

It deliberately ignores two things that a naive regex reports as broken, both of
which are really in the documents:

- **code spans.** `` `[int](3/2)` `` is PowerShell in a results table — the
  banker's-rounding bug that mislabelled every early sweep column "median" — and
  it appears in four reports.
- **`sandbox:` and absolute URLs**, which arrive pasted inside external research
  replies under `docs/researchs/`.

Percent-escapes are decoded before the check, because `Deep%20Research/` is a
real directory and a markdown renderer resolves it.

Last clean run: **2026-08-21, 66 files, 84 links, 0 broken.**

---

## `audit-stale-claims.py`

```powershell
python C:\AI\scripts\audit-stale-claims.py          # the worklist
python C:\AI\scripts\audit-stale-claims.py --ids    # the rules only
```

Every rule is a claim this project **published and then contradicted with its
own data**. The failure it guards against is specific and has already happened
here more than once: a corrected report and an uncorrected one both exist, an
agent reads the wrong one, and the correction may as well not have been written.

The register the rules point at is
[`../docs/reports/CORRECTIONS.md`](../docs/reports/CORRECTIONS.md).

**A hit is not a defect.** A report that describes a retraction matches its own
rule — `24-BEYOND-128K.md` matches four of them, because it is where four of the
corrections were written. The output is a list of lines to read, not a verdict,
and it exits 0 either way. It is a reading aid, not a gate: a gate on prose
would either block on its own corrections or be tuned until it never fires.

**Adding a rule is part of retracting a claim.** When a measurement contradicts
something already written, the retraction is not finished until the claim has an
entry here and in `CORRECTIONS.md` — otherwise the next sweep will not find the
copies you missed.

---

## The docs gate

Both scripts, before trusting the documentation:

```powershell
python C:\AI\scripts\check-doc-links.py ; python C:\AI\scripts\audit-stale-claims.py
```

Last clean: **2026-08-21 — 71 files, 106 links, 0 broken.** The audit is not
expected to reach zero and should not be driven there.
