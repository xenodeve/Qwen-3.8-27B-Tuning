## ⛔ Two things before any of the below

Both are measured failures of this suite used on its own — 2026-09-08, five runs of the same
Thai page brief (`Qwen-3.8-27B-Tuning/docs/results/14-frontend-arms-2026-09-08.md`).

**1. Write the file first, then describe it.** Create the HTML file with real content as your
**first** artifact action — before the aesthetic directions, before the LIFT plan, before any
summary of what you are about to build. Elaborate by editing the file that already exists.

> *In 2 of 5 runs this suite produced **no file at all**. Both exited `rc = 0` after 19 and
> 22.7 minutes, having planned the design in prose and stopped mid-sentence on
> "สร้างไฟล์เดียว:" — "create a single file:". Nothing was ever written. The arms carrying
> `design-ship-gate` never did this.*

**2. The page copy is in the brief's language.** If the brief is Thai, the headline, body,
buttons and nav are Thai. English stays only in identifiers, code and brand names. Check
before you finish:

```sh
grep -c "[ก-๛]" page.html      # a Thai brief: must be well above 0
```

> *This suite mentions language nowhere — the rule lived only in `design-ship-gate`. Used
> alone it answered a Thai brief in **English in 3 of 3 pages it produced**, while every arm
> carrying the gate wrote 1,300–2,000 Thai characters.*

---
