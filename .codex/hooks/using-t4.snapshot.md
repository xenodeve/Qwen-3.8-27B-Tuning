---
name: using-t4
description: Use at the start of any task in a T4-team repo (T4 Labs / Slow-Inc) to decide which T4 skill applies — setting up the repo, session-start memory and open-work, planning or shipping a feature, or recording a bug/decision/change — and to hand off to the companion skill ecosystems the team runs alongside (superpowers, matt pocock, 9arm). The entry-point map for the t4-* skill family; consult it before acting when you're unsure which T4 discipline or which external skill a task needs. Triggers include working in a repo whose CLAUDE.md points here, "how do we work in this repo", "which T4 skill", and starting a session in a Slow-Inc repo (MangaDock, T4-Fastwork).
---

# Using T4

## Overview

The T4 team runs its repos **agent-primary** — the coding agent is the main developer, so the repo's docs are its operating manual, not team paperwork. The `t4-*` skills encode that standard; this skill is the **map** telling you which to invoke.

**Core rule:** when a task matches a skill below, **invoke it before acting** — don't work from memory; skills evolve. (User instructions — `CLAUDE.md` / direct requests — always win.)

## Route first — before you respond

Before you answer, ask a clarifying question, explore, edit, or run any tool: check the map below and invoke the matching skill. **Uncertainty is a reason to consult the map, not skip it** — but a leaf skill fires on its own explicit trigger, not a hunch (don't invoke `security-review` because a line contains "token"). Announce **"Using `<skill>` to `<purpose>`"**, then invoke — announcing is not the work.

**Re-route at every phase boundary.** A check at task start does **not** discharge a later trigger: after writing code → `simplify`; before merge → `code-review` + `scrutinize`; touched auth/secret → `security-review`; implementation done → `verify`. A parent skill (`t4-dev-workflow`) **does not discharge** its leaves (`tdd`, `verify`, …).

**Red flags — these thoughts mean STOP and route:**

| Thought | Reality |
|---|---|
| "Small change, skip it" | Small changes cross phases too. Route. |
| "I know the T4 workflow" | Skills evolve — load the current one. |
| "Tests already exist, so it's TDD" | Existing tests ≠ red-first for this change. Load `tdd`. |
| "t4-dev-workflow covers it" | Parent ≠ leaf; invoke the phase skill. |
| "Just a token/config value" | If it plausibly crosses a trust boundary → `security-review`. |
| "I announced it, that's enough" | Announcing ≠ doing. Invoke and follow. |
| "Obviously it's X — just fix it" | Obvious ≠ traced. Find the root cause first. |
| "It should work — call it fixed" | Should ≠ does. Name the evidence or hedge. |
| "Let me look around first" | Route first — the skill tells you HOW to look. |

## The map — route by what you're doing

| You are… | Invoke |
|---|---|
| **Starting a session** — where work left off, what's open, what past sessions decided | **`t4-agent-memory`** (`Home.md` → ledger → the issue) |
| **Setting up a new repo**, or retrofitting one missing the operating layer | **`t4-project-bootstrap`** |
| **Planning or building a feature** — an idea to ship, filing an issue/PRD, writing a bilingual issue/PR body, opening a PR | **`t4-dev-workflow`** |
| **Something notable just happened** — fixed & validated a bug, made a hard-to-reverse decision, shipped a system-affecting change | **`t4-engineering-records`** |
| **Recording or recalling durable memory** — persisting a convention/decision/feedback, or finding where open work lives | **`t4-agent-memory`** |
| **Going AFK** — the developer hands you a bounded batch to run unattended and steps away ("handle it", "clear the queue", "keep going without me") | **`t4-afk`** |

## Companion ecosystems — use them alongside T4

The `t4-*` skills sit **on top of** three general ecosystems — not alternatives. Invoke them by their own triggers; every slash-command a T4 skill names (`/grill-me`, `/to-prd`, `/tdd`, `/debug-mantra`, `/post-mortem`, `/scrutinize`) lives in one of these.

| Ecosystem | Reach it via | Use it for | Representative skills |
|---|---|---|---|
| **superpowers** | its own map — invoke **`superpowers:using-superpowers`** first | general process discipline (*how to work*) | `brainstorming`, `test-driven-development`, `systematic-debugging`, `verification-before-completion` |
| **matt pocock** (`mattpocock/skills`) | `/setup-matt-pocock-skills` (installs + configures tracker/labels/domain layout) | **the flow the T4 pipeline is built on** — grill→spec→tickets, plus the tracker/label/domain-doc conventions T4 reuses | `grilling` (`/grill-me`), `to-prd`, `to-issues`, `domain-modeling`, `code-review` |
| **9arm** (`thananon/9arm-skills`) | `npx skills add thananon/9arm-skills` | debugging + adversarial review discipline + cheap delegation | `debug-mantra`, `post-mortem`, `scrutinize`, `qwen-agent` |

**Routing rule:** T4 skills own the *team-specific* decision (which record, which memory layer, the tracker rules) and **hand off the technique** to the ecosystem skill — `t4-engineering-records` decides a bug needs a post-mortem, then invokes `/post-mortem`. For non-T4 *how to work*, prefer `superpowers:using-superpowers`.

## Session-start protocol

At the start of any session in a T4 repo, before picking up work:

1. **`karpathy-guidelines`** — load once at session start so every edit this session is surgical, simple, and goal-verified (see the coding-behavior rule below).
2. **`t4-agent-memory`** — read the vault index + open-work ledger (what survives a context reset), then the issue you're picking up.
3. Route the task through the map above.

## The non-negotiable rules (all skills carry these)

**Evidence before verdict, fix, or exemption** — three faces of one rule; each is detailed in `t4-dev-workflow`:

- **Don't state as settled what you haven't verified.** *Fixed · works · passes · safe · done · the root cause is* each need the command you ran, its output, or the `file:line` you read, named with them. Otherwise label it a **hypothesis** — and a claim's register never improves by being repeated. "Tests not run" is a complete sentence.
- **Root cause before fix** — reproduce, trace the real failing path, falsify, *then* propose (`/debug-mantra`). An untraced fix is a guess.
- **Skipping a rule requires proof, not judgment.** The default is comply. State a **checkable fact about this change** a reviewer can verify without redoing your reasoning (e.g. "`git diff --name-only` is `*.md` only"). "Small", "obvious", "unrelated", "slow", "they're in a hurry" are **not** proofs. **No proof → follow the skill.** An unstated skip is a violation; write the exemption where the work is reported. Hook-enforced and safety rules are never exemptable by argument.

- **Memory is first-class** — record what you ship; the next agent inherits only what you wrote (`t4-agent-memory`).
- **PRD → issues → PR** — never a PR without a referenced issue; issues are the source of truth (`t4-dev-workflow`).
- **Bilingual is tracker-only, Thai mirrors English exactly** — issue/PRD/PR bodies; not chat/reports; identifiers stay English (`t4-dev-workflow`).
- **TDD is mandatory**; **verify every frontend change end-to-end** (unit tests can't see real layout/hydration).
- **Non-standard framework version → read the vendored docs first**, not memory.
- **Bun** is the default package manager (`bun.lock`, `bunx`).
- **Records stay a reliable index** — `file:line`, commit SHAs, validated-only (`t4-engineering-records`).
- **Glossary is load-bearing**; **proceed silently if a governance file is absent**.
- **Coding behavior follows `karpathy-guidelines`** — simplest thing that works, surgical diffs tracing to the request, verifiable success criteria.
- **Some rules are machine-enforced** — the `PreToolUse` gate + a git pre-push guard block a PR with no issue, dangerous git, a failed `verify` (`t4-dev-workflow`). They raise the floor, not replace judgment.
- **Act on what's already decided; don't re-ask.** If a standing instruction, the tracker (label / issue body / ledger), or your own earlier recommendation answers it, *act*. Interrupt only for a genuinely unresolved decision that's the developer's — prefer parking + one digest. Re-asking what you can answer yourself is the "sticking" anti-pattern (`t4-afk`).

## When NOT to use

A non-T4 project (the bilingual rule, labels, and memory layout are team-specific), or a throwaway prototype with no issues/memory — use the general skills directly.
