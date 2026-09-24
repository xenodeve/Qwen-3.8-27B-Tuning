# Sharp, Swift and TURBO evaluation extension

Status: executing under issue 92. No production default change is authorized.
The permanent selection rule now lives in `AGENTS.md` under the engineering
north star; this plan applies that rule rather than redefining it.

## Question

Extend the completed GSQ/NVFP4/EXL3 comparison with three distinct interventions:

1. Sharp v22.5.0 template on the existing NVFP4 weights.
2. UkisAI Swift post-trained weights at Q6_K.
3. DavidAU TURBO post-trained/merged weights at MTP-Q6_K.

Rank neither raw decode nor response brevity alone. The primary criterion remains
time to verified completion together with correctness, requirement retention,
language integrity and stability. Full visible thinking and unsuccessful costs
remain evidence.

## Frozen upstream identities

- Sharp: `peculiar-ragdoll/Qwen-Sharp-Chat-Templates`, revision
  `85461fc118aaf25e7319c7ecf2481f944aac3a32`, `chat_template.jinja`, embedded
  version `qwen3.8-froggeric-v22.5.0`, SHA256
  `cdff39fb26b60dc90faa292e726655c6b21f62db497846e02e4c4bbab942a84a`.
- Swift: `ukisai/Swift-Qwen3.8-27B-GGUF`, revision
  `eb0e3a7dc70643c2ab920f5133750d02c20849ff`,
  `Swift-Qwen3.8-27B-Q6_K.gguf`, 22884407456 bytes, Hub LFS SHA256
  `7f4de8abd5446c08b0f975a1b38e43d02639f4ae9ecdd2ddfd5c5b0f612bda59`.
  The repository's downloaded `SHA256SUMS.quants` instead says `a429f636...`;
  record the mismatch and verify the object selected by the pinned revision.
- TURBO: `DavidAU/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NEO-CODER-MAX-MTP-GGUF`,
  revision `c02caef111a8acf987947f35e1e288aa5450e184`, MTP-Q6_K artifact,
  24033703456 bytes, Hub LFS SHA256
  `ac011aabe685edbdf542e49351eb6c76c0e5531408f2507f2235ab10931e23a5`.

These are pinned local artifacts. Creator/community benchmark claims are external
claims, not rows in the local result table.

## Isolation and sequence

1. Verify baseline suite, disk, exact files and checksums.
2. Add explicit `artifact`, `template` and `effort` dimensions to the existing
   experimental runner test-first. Defaults remain NVFP4/Stock outside the runner.
3. Sharp 2x2: same NVFP4 artifact/config/context/sampler/speculation under Stock
   versus Sharp and medium versus xhigh. Template and effort are independent.
4. Swift and TURBO: first run Q6_K with Stock embedded template, medium effort and
   speculation off to isolate weight behavior. Then tune each passing artifact locally;
   the final comparison uses each artifact's own best measured configuration rather
   than forcing one inherited recipe.
5. Use one common allocated context. Start at 147456; if a Q6 arm cannot fully
   offload and serve the workload, reduce the allocation for every new arm and
   record the incompatibility. The final common window is the largest allocation no
   greater than the smallest context every candidate can actually serve. Never
   compare silently truncated histories.
6. Retain the original-session replay and held-out executable coding tasks from
   result 20. Add math characterization for Swift and explicit completion,
   continuation, duplicate-work, tool-error and reasoning-loop counters for TURBO.
   Preserve full visible thinking and charge thinking, answer, tools, verification,
   repair, retry and failed attempts to time per verified task. Report prefill and
   decode as explanatory components, not standalone rankings.
7. Measure repairable defects such as language leakage both raw and with candidate
   guards; charge guard latency and false changes, and never upgrade a failed or
   truncated task through repair. Choose the strongest foundation before further
   optimization.
8. Rotate order over at least three validation rounds for any promotion verdict.
   Preserve load failures and spent time. Run one inference model at a time.

## Initial gates

- Artifact and template identity are verified in each manifest.
- Runtime context equals requested context; listener belongs to the spawned tree;
  no CPU layer spill is accepted as a matched speed result.
- Code is executed against frozen assertions. A correct draft in thinking does not
  upgrade a failed or missing final answer.
- Sharp changes only the template; it does not silently alter effort or weights.
- Thai/mixed-language output is manually reviewed alongside deterministic script
  counts. Han masking is a mitigation, not proof of correct language.
- Full repository tests pass after integration. Existing unrelated dirty files are
  preserved. No launcher/default is promoted automatically.

## Environment note

The driver now reports 616.92; result 20 recorded 616.64. New candidate rounds
must be paired within this campaign. Do not interpret a raw speed difference
against result 20 as an artifact-only effect across the driver change.

## Scope boundary

The first campaign establishes compatibility and bounded behavioral evidence. A
daily-driver promotion still requires broader multi-file agent tasks, tool-error
recovery and long-horizon/soak evidence. TURBO specifically cannot pass its final
gate on short single-response code tasks alone.
