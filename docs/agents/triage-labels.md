# Triage labels

Five roles, plus Type / Component / Severity. The five triage names are reused
unchanged from the matt pocock ecosystem; the three groups below are the T4
delta.

## Triage role — exactly one per issue

| label | means |
|---|---|
| `needs-triage` | nobody has decided what this is yet. **The honest default** — assigning a role to an unanalysed issue is worse than admitting it is unanalysed |
| `ready-for-agent` | scoped enough that an agent can start without asking |
| `ready-for-human` | needs a decision only the developer can make |
| `blocked` | waiting on something we do not have |
| `wontfix` | decided against, with the reason in the issue |

## Type

`bug` · `feature` · `experiment` · `instrument-fault` · `docs` · `chore`

**`instrument-fault` is specific to this repo** and is not a `bug`: it is a
defect in the measuring apparatus that produced a believable number instead of a
failure. Thirteen are documented. Keeping them separate is what makes the rate
readable.

## Component

`bench` · `harness` · `docs` · `ci` · `serving` · `model`

## Severity

`critical` · `Major` · `minor`

**A `security` issue must be `critical` or `Major`.** No exceptions.

## Creating them

`gh label create <name> --repo xenodeve/Qwen-3.8-27B-Tuning --color <hex>
--description <text>`

**A documented vocabulary with no labels behind it is the failure this file
exists to prevent.** Measured elsewhere in this family: a bootstrapped repo had
8 of 19 documented labels actually created, `needs-triage` among the missing.
Report which were created, which already existed, and which were skipped.
