# Qwen3.8-27B — Q3 vs Q4 optimization results

## Environment caveat

Free VRAM before load across 22 recorded launches ranged **9933–10530 MiB**. `--fit on` derives the layer split from whatever is free at boot, so runs from different boots are not directly comparable. Comparisons below are within-sweep.

## Speculation matrix (synthetic decode)

### UD-Q4_K_XL

| spec | n_max | prompt | tok/s median | min–max | acceptance | VRAM free |
|---|---|---|---|---|---|---|
| none | — | bench | **8.24** | 8.22–8.24 | — | 450 MiB |
| none | — | code | **8.22** | 8.2–8.25 | — | 450 MiB |
| ngram-simple | 4 | bench | **8.29** | 8.21–8.3 | — | 509 MiB |
| ngram-simple | 4 | code | **8.37** | 8.17–8.72 | 30.8% | 509 MiB |
| draft-mtp | 2 | bench | **10.67** | 10.58–10.78 | 78.1% | 888 MiB |
| draft-mtp | 2 | code | **12.1** | 12.03–12.22 | 98% | 888 MiB |
| draft-mtp | 3 | bench | **9.91** | 9.49–11.13 | 70.3% | 1034 MiB |
| draft-mtp | 3 | code | **12.03** | 11.1–12.58 | 88.8% | 1034 MiB |

### UD-Q3_K_XL

| spec | n_max | prompt | tok/s median | min–max | acceptance | VRAM free |
|---|---|---|---|---|---|---|
| none | — | bench | **9.01** | 9.0–9.29 | — | 471 MiB |
| none | — | code | **9.25** | 9.19–9.27 | — | 471 MiB |
| ngram-simple | 4 | bench | **9.16** | 9.14–9.16 | — | 505 MiB |
| ngram-simple | 4 | code | **9.08** | 9.07–9.09 | 30.8% | 505 MiB |
| draft-mtp | 2 | bench | **8.88** | 8.66–8.96 | 77.5% | 723 MiB |
| draft-mtp | 2 | code | **10.3** | 9.86–10.32 | 96.4% | 723 MiB |
| draft-mtp | 3 | bench | **7.27** | 7.04–7.73 | 64.1% | 1063 MiB |
| draft-mtp | 3 | code | **9.92** | 9.89–9.96 | 99.1% | 1063 MiB |

## Quality benchmark (verified by execution)

| config | pass rate | verified tasks/hr | median tok/s | wall | temp | effort |
|---|---|---|---|---|---|---|
| smoke-q4-mtp2 | **80.0%** (8/10) | **29.5** | 11.18 | 976.0s | 1.0 | medium |
| q4-draft-mtp2-t1.0 | **90.0%** (27/30) | **33.6** | 10.56 | 2889.5s | 1.0 | medium |
| q3-draft-mtp2-t1.0 | **86.7%** (26/30) | **22.2** | 8.73 | 4213.0s | 1.0 | medium |

### Per-task pass counts

| task | difficulty | smoke-q4-mtp2 | q4-draft-mtp2-t1.0 | q3-draft-mtp2-t1.0 |
|---|---|---|---|---|
| `bracket_matching` | easy | 0/1 | 0/3 | 0/3 |
| `lru_cache` | easy | 1/1 | 3/3 | 3/3 |
| `merge_intervals` | easy | 1/1 | 3/3 | 3/3 |
| `damerau` | hard | 1/1 | 3/3 | 3/3 |
| `lfu_cache` | hard | 1/1 | 3/3 | 2/3 |
| `tree_codec` | hard | 1/1 | 3/3 | 3/3 |
| `expr_eval` | medium | 0/1 | 3/3 | 3/3 |
| `rotated_search` | medium | 1/1 | 3/3 | 3/3 |
| `text_wrap` | medium | 1/1 | 3/3 | 3/3 |
| `toposort` | medium | 1/1 | 3/3 | 3/3 |

