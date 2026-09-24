# Thai LLM Output Repair Algorithms — Research (2026-09-15)

## Question
Can a lightweight algorithm repair malformed Thai output after the primary LLM, before adding a second repair model? Target failures are duplicated Thai vowels/tone marks, malformed combining-mark order, known recurring misspellings, and repetition loops. Code comments, code, JSON, paths, URLs, and tool arguments must not be silently rewritten.

## Findings

### 1. PyThaiNLP deterministic normalizer — best first candidate
Primary source: https://github.com/PyThaiNLP/pythainlp/blob/dev/pythainlp/util/normalize.py

Relevant APIs:
- `reorder_vowels(text)` — puts Thai vowels/tone marks into standard logical order; source example includes `เเปลก -> แปลก`.
- `remove_repeat_vowels(text)` — removes repeating vowels, tone marks, and signs. Source examples: `นานาาา -> นานา`, `ดีีีี -> ดี`.
- `remove_dangling(text)` — removes dangling non-base marks.
- `normalize(text)` — composes the above with zero-width-space removal, duplicate-space cleanup, mark ordering, repeat-mark removal, and dangling-mark removal.

Assessment: high-confidence deterministic repair for the exact duplicate-mark class; CPU-only and lightweight. Do not run blindly over code/JSON/tool payloads.

### 2. Unicode NFC — useful but not a repairer
Sources:
- https://www.unicode.org/reports/tr15/
- https://docs.python.org/3/library/unicodedata.html

`unicodedata.normalize("NFC", text)` canonicalizes Unicode representation but does not collapse repeated Thai marks. Tested examples such as `ก่่`, `นานาาา`, and `ดีีีี` remain repeated. Use only as a canonicalization step, not as the Thai repair algorithm.

### 3. Known-broken-pattern dictionary — cheapest and safest for recurring defects
Use an allowlisted mapping learned from verified transcripts, e.g. a known malformed form → verified intended form. Complexity is effectively O(1) per lookup and no model/GPU is needed.

Guardrails:
- only exact known patterns;
- only within a detected Thai prose span;
- never match inside code, JSON, URLs, paths, identifiers, tool calls, or fenced/inline code;
- every replacement must have an eval pair and false-positive test.

A dictionary has a hard recall ceiling: it cannot repair a new malformed word. Corpus evidence showed a long Thai lexical tail (many hapax words), so it is an arm, not the whole solution.

### 4. PyThaiNLP spell correction — second lightweight arm, higher risk
Sources:
- https://github.com/PyThaiNLP/pythainlp/blob/dev/pythainlp/spell/core.py
- https://github.com/PyThaiNLP/pythainlp/blob/dev/pythainlp/spell/wanchanberta_thai_grammarly.py

`spell()`/`correct()` can perform dictionary/model-based spelling correction. This is semantic correction, not mark deduplication. It may help dictionary-covered malformed words but can produce false positives on names, technical terms, transliterations, source code comments, and new words. It needs a paired eval with precision/recall and code-preservation checks before production use.

### 5. Edit-distance dictionary (SymSpell/BK-tree) — useful only with a controlled vocabulary
A SymSpell/BK-tree can find near words cheaply, but Thai has no whitespace word boundaries in the general case and a low edit distance is not proof of intended spelling. It is suitable after Thai word segmentation and only with a confidence threshold plus an allowlisted vocabulary. It should not be allowed to rewrite arbitrary OOV output.

### 6. Thai word segmentation + lexicon/Viterbi repair — plausible algorithmic path
A stronger non-neural design is:
1. segment a Thai prose span;
2. detect malformed mark runs deterministically;
3. generate dictionary candidates using weighted edit distance;
4. score candidate segmentations with a Thai lexicon/frequency model;
5. repair only above a strict confidence margin;
6. otherwise return raw output or route the span to a second model.

This can cover new defects better than exact mapping while remaining CPU-only, but it is more code and needs a real corrupted/clean pair benchmark. OOV alone is unsafe because names, jargon, compounds, and transliterations are common.

### 7. Second model — fallback, not first line
A small Thai correction model (for example WanchanBERTa-backed correction in PyThaiNLP, or a small Typhoon model) can repair semantic/new-word errors that algorithms cannot. Cost is CPU/RAM latency and it can hallucinate or alter meaning. Invoke only when deterministic repair detects a defect and only on the affected prose span; never send code/tool arguments through it.

### 8. Loop detector — complementary, not a repairer
The existing EXL3 `loop_guard.py` detects runaway repeated output during generation and cancels the job. It prevents thousands of bad characters but does not reconstruct a complete final answer. It should remain separate from post-generation Thai repair.

## Recommended pipeline for EXL3 profile G

```text
stream output
  -> LoopGuard during generation (stop runaway loop)
  -> segment output: prose vs fenced/inline code, JSON, tool calls, URLs, paths
  -> NFC only for eligible prose spans
  -> PyThaiNLP remove_repeat_vowels/reorder_vowels on eligible Thai prose
  -> exact verified broken-pattern dictionary
  -> validate changed span and compare raw vs repaired
  -> if low confidence/new spelling defect: keep raw or optional second-model fallback
```

For a first experiment, use only `remove_repeat_vowels()`/`reorder_vowels()` plus the existing verified dictionary, and record:
- raw output;
- repaired output;
- changed spans;
- repair reason;
- false positives on clean Thai prose;
- unchanged code/JSON/tool/path regions;
- CPU latency.

## Verdict

There is already a good lightweight implementation for duplicated Thai vowels/tone marks: **PyThaiNLP's deterministic normalizer**. No evidence found of a universal existing filter that safely repairs every malformed Thai LLM output while preserving code and tool payloads. The best architecture is deterministic repair first, constrained dictionary/edit-distance second, and a small second model only as a gated fallback. NFC alone is insufficient; OOV alone is unsafe.
