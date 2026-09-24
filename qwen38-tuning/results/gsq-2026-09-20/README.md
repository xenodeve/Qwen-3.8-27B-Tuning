# GSQ comparison evidence (initial screen)

The language-only stop below is historical and superseded by the user's next
directive. Evaluation resumed under time per verified task plus quality, at a
common allocated context with per-artifact best-known recipes and guard trials.
New campaign output: ../gsq-time-quality-2026-09-20/.

Tracking: https://github.com/xenodeve/Qwen-3.8-27B-Tuning/issues/92

These are exploratory observations, not a promotion verdict or paired speed claim.
No current default was changed. Server processes are run one at a time.
The initial Thai gate failed. Expanded tuning and optional tracks were held at
that gate; the full integration plan was not completed.

## Environment and provenance

- Windows 10.0.26200, NVIDIA driver 616.64.
- RTX 4070 SUPER 12 GB + RTX 5060 Ti 16 GB, PXB topology.
- Installed RAM reported as 50,008,496 KiB.
- llama.cpp binary: C:/AI/llama.cpp-blackwell/llama-server.exe,
  build 10499, source 1deefcca395743049c3820ab8f9b15043f3e9446.
- EXL3 fork: C:/AI/exllamav3-mia,
  commit 63b32f001d7b2cfed3b3e3aaf25f534ba53cc7ed.
  It already has a local modification in exllamav3/modules/attention_fn/bc_dsa.py;
  the retained diff is exl3-fork-local.patch and the file SHA256 is
  281a2260e452142471d07f4744eee601a799811cc55a9f6a2162b8d8d42f235b.
  This session did not edit that fork file.
- Model sizes, SHA256, argv and per-GPU snapshots are recorded in each manifest.
- GSQ source revision d562806dbafae37109975e970aae91b43e73b440;
  downloaded size and SHA256 verified against upstream before loading.

## Baseline gate

Initial suite: 1708 passed, 18 failed, 2 skipped. CUDA_PATH was absent for 17
Unsloth-build preview checks; CRLF checkout altered the frozen corpus hash for
one. Restoring LF and setting CUDA_PATH for tests yielded 1726 passed, 2 skipped
(baseline-tests.xml). New comparison tests have separate red/green evidence in
the session; these were not included in the initial suite collection.

## Original-session replay limitations

Source session: 096ebcae-a6ae-4b80-b608-3b60f60a3fac in D:/Github/YT Downloader.
Claude Code resumed it with --fork-session --no-session-persistence into a local
capture endpoint. No model ran during capture and no tool calls were returned.
The source SHA256 before and after is identical (session-capture/provenance.json).
The captured request retains 56 messages and 27 tools; no synthetic filler or
other session was inserted. Current system/tool definitions may differ from the
incident. MCP was disabled. The capture endpoint's token-count stub is NOT evidence.

The historical visibly corrupted message msg_3d8ae2c199ef reports 10844 new input
+ 143104 cached input = 153948 tokens. The current request is not identical:
EXL3's count endpoint reports 140726 for the captured settings; the actual
generation request with medium effort reports 140688. NVFP4's template reports
138942 for the converted same conversation. Do not call these 143.2k or 153948.
They are original-session replays with their measured depths, not exact incident
reproductions. None meets the runner's incident_depth_reached threshold of 143200.

## Invalid and exploratory observations

- The initial bracket_matching case cannot be scored: its seventh fixture
  assertion contains characters [40,39,92,39,41], where the closing quote is
  escaped, but expects True. Both incumbents return False. Keep the raw failures;
  they are fixture-invalid, not demonstrated model regressions. The subsequent
  screen uses frozen toposort for every candidate.
- 20260920-013107-nvfp4-served-c200704-r1 stopped before inference because the
  production verbosity omitted required architecture evidence. The next run
  raises only log verbosity to 5 and records the resolved argv.
- NVFP4 production flags come from the actual worker's -WhatIf output, not the
  arena's older nvfp4-served snapshot (which differs in current cache/n-gram flags).
- NVFP4 at 200704 is an extra deep-window probe; the normal worker window is
  147456 and needs its own reference row.
- The initial incumbent runs overlapped model downloading/hashing on the CPU and
  disk. Their timing values are retained, but are not controlled speed comparisons.
- Residency samples without --while-generating have under_load=False in the old
  sampler schema because that flag means the sampler launches its own request.
  Here requests were already running in gsq_compare; their server logs provide
  concurrent load evidence. Shared GPU memory alone does not prove eviction.

## Observed quality so far

- GSQ without speculation passed merge_intervals and toposort executable tests.
- GSQ short Thai prose includes Chinese "确认" in "เพื่อ确认ว่า". This is a
  directly observed language leak; the prose is not clean Thai.
  The same output recurred in the final fresh-boot smoke check with seed 42.
  A repeated seed is a reproduction check, not two independent rate samples.
- NVFP4 deep replay includes "เมือรัน" and "แก้มคำเพี้ยน". These are direct
  orthographic errors in the returned text.
- EXL3 deep replay finished without a loop and has awkward phrasing such as
  "ความเข้าได้ถึงการใช้งาน". Its production Han ban and Thai temperature cap
  remain active. Comparing Han counts against unguarded GSQ does not isolate
  artifact quality.
- All three can produce a correct repair of the short supplied sentence; that
  does not establish spontaneous Thai generation reliability.

## Still not established

No statistical ranking, calibrated Thai error rate, tuned GSQ draft depth,
certified context ceiling, full agent-quality gate, vision test, or soak verdict
has been established by these initial screens. Raw response files contain full
requests and generated text. Treat session-derived evidence as local/private;
review before publishing any captured request.
