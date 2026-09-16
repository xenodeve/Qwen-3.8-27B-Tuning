# System impact register

One line per system-affecting change, newest on top. **Pointers, not copies** —
the full record lives in the file named, and duplicating it here is how two
versions drift apart.

| date | change | record |
|---|---|---|
| 2026-09-16 | Retracted the Thai repair `40 -> 0, FP 0` verdict: the substring dictionary corrupted correct Thai while its eleven-pattern counter could not observe the damage. Replaced it with an expected-text oracle, a precision-first offline subset, protected spans, and a regression that keeps valid Thai mark continuation tokens available | [post-mortem](2026-09-16-thai-repair-fp-oracle.md) |
| 2026-08-21 | Retracted the claim that the model loops in its reasoning; the probe now keeps the full trace and scores its repetition. Suite 108 -> 111 | [post-mortem](2026-08-21-inferred-looping-from-three-numbers.md) |
