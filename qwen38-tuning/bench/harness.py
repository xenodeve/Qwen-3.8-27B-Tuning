"""Measurement primitives for the tuning sweeps.

Every function here replaces one that failed silently earlier in this project.
Silence is the shared failure mode: a wrong median labelled "median", a dropped
row that left a table looking complete, a layer count that did not add up. So
these raise rather than guess.
"""


def median(samples):
    """True median. Raises on empty input rather than inventing a number."""
    if not samples:
        raise ValueError("median of no samples")
    s = sorted(samples)
    mid = len(s) // 2
    if len(s) % 2:
        return s[mid]
    return (s[mid - 1] + s[mid]) / 2


def load_jsonl(path):
    """Read a results file. utf-8-sig strips the BOM PowerShell writes.

    A malformed line raises with its line number instead of being skipped:
    the earlier `except JSONDecodeError: pass` is what let the BOM quietly
    delete the baseline row from every table.
    """
    import json
    from pathlib import Path

    rows = []
    with Path(path).open(encoding="utf-8-sig") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path} line {lineno} is not valid JSON: {e}") from e
    return rows


_LAYER_RE = None


def _assignment_passes(log_text):
    """The layer-assignment lines, split into passes.

    llama.cpp emits several reserve passes and each restarts at layer 0, so a
    pass boundary is simply where the index stops increasing. Shared by
    `parse_layer_split` and `target_layer_count` so the two can never disagree
    about where one pass ends and the next begins.
    """
    global _LAYER_RE
    if _LAYER_RE is None:
        import re
        _LAYER_RE = re.compile(
            r"load_tensors: layer\s+(\d+) assigned to device (\w+)")

    pairs = _LAYER_RE.findall(log_text)
    if not pairs:
        raise ValueError("no layer-assignment lines found; was -lv 5 passed?")

    devices = [d for _, d in pairs]
    idx = [int(i) for i, _ in pairs]
    bounds = [0] + [i for i in range(1, len(idx)) if idx[i] <= idx[i - 1]] + [len(idx)]
    return [devices[a:b] for a, b in zip(bounds, bounds[1:])]


def parse_layer_split(log_text, total=None, expect_layers=None):
    r"""Count GPU vs CPU layer placement from a verbose llama.cpp load report.

    Counts the FINAL assignment pass, found from the layer indices themselves.
    llama.cpp emits several reserve passes -- 451 lines for a 40-layer MoE --
    and each pass restarts at layer 0, so a pass boundary is simply the point
    where the index stops increasing.

    This replaces a hardcoded `total=65`, which was Qwen3.8-27B's 64 blocks plus
    its MTP head and silently wrong for every other architecture. On the
    35B-A3B MoE it reported "65 + 0" for a 41-layer model by slicing the last 65
    lines across two passes. Those arms happened to be fully resident so the
    conclusion held, but it held by luck: the same slice would have reported
    65+0 with layers on the CPU. `block_count` is not a usable substitute
    either -- Qwen3.8 logs 65 for 65 layers, the MoE logs 40 for 41.

    `total` is kept only as a fallback for logs whose indices cannot be read.

    `expect_layers` names WHICH model you mean, and is required once a draft
    model is loaded. Such a log carries several models' passes -- with the
    DFlash2 drafter the order is drafter(6), target(65), target reserve(65),
    drafter(6) -- so "the last pass" is the drafter, and this returned (6, 0)
    for a 65-layer target on 2026-08-22 (issue #17). That is a healthy-looking
    split describing the wrong model, in which a spill of the target could never
    appear. Pass the target's layer count and the last pass of that size is
    used; a size no pass has raises rather than falling back to another pass.

    (\w+) rather than (\S+) because the device token carries a trailing comma
    ("CUDA0,"), which made an exact == "CPU" comparison match nothing.
    """
    passes = _assignment_passes(log_text)
    devices = [d for p in passes for d in p]

    if expect_layers is not None:
        matching = [p for p in passes if len(p) == expect_layers]
        if not matching:
            raise ValueError(
                f"no assignment pass has {expect_layers} layers; "
                f"passes seen: {[len(p) for p in passes]}"
            )
        last = matching[-1]
    else:
        last = passes[-1] if total is None else devices[-total:]
    gpu = sum(1 for d in last if d.startswith("CUDA"))
    cpu = sum(1 for d in last if d == "CPU")
    if gpu + cpu != len(last):
        raise ValueError(
            f"layer split {gpu}+{cpu} does not account for {len(last)} lines; "
            f"unexpected devices: {sorted(set(last) - {'CPU'} - {d for d in last if d.startswith('CUDA')})}"
        )
    return gpu, cpu


def target_layer_count(log_text):
    """How many layers the TARGET model has, read from its own load report.

    Replaces a caller-side constant. `dflash2_arena` hardcoded 65 -- "64 blocks
    plus the MTP head" -- which was the count for `UD-IQ2_XXS`, an artifact with
    no MTP head at all. `UD-Q2_K_XL` has one at `blk.64` and reports 66, so
    every row it produced raised instead of recording a split (issue #44).

    THE TARGET IS THE MODEL WITH THE MOST LAYERS. True of every configuration
    this project runs: the DFlash2 drafter is 6 against a target's 65 or 66, and
    with `draft-mtp` there is no second model at all. If a drafter ever carries
    more layers than its target this inverts -- stated here rather than assumed,
    because a wrong answer would look like a healthy split for the wrong model,
    which is the fault `expect_layers` was added to prevent in the first place.

    Raises on a log with no assignment lines rather than returning a default:
    the caller feeds this straight into `parse_layer_split`, and a guessed count
    there is exactly the silent fallback that function refuses.
    """
    passes = _assignment_passes(log_text)
    return max(len(p) for p in passes)


def project_prefill_seconds(pp_tok_s, ctx_tokens):
    """Straight-line cold-prefill estimate. Ignores depth degradation by design."""
    if pp_tok_s <= 0:
        raise ValueError(f"prompt-processing rate must be positive, got {pp_tok_s}")
    return ctx_tokens / pp_tok_s


# The harness fills ~80 % of the window before timing anything; see
# depth_sweep.run(). The budget has to cover that prefill, not the window.
PREFILL_FILL_FRACTION = 0.8

# Slowest cold prefill ever measured on a RESIDENT arm at 131,072 is 240.6
# tok/s (AD-IQ1_M at 65+1). The slowest PATHOLOGICAL one is 8.56 tok/s
# (`ot-ffn-1`, 644 MiB of FFN weights forced to CPU). 60 sits four times below
# the legitimate worst case and seven times above the pathological one, so the
# budget cannot truncate a real measurement and cannot sit an hour on a dead
# arm. It is a floor, not an expectation.
PREFILL_FLOOR_TOK_S = 60.0

# Load, health poll, the five 160-token generations and the greedy probe.
TIMEOUT_MARGIN_S = 300


def completion_timeout_s(ctx, floor_tok_s=PREFILL_FLOOR_TOK_S,
                         margin_s=TIMEOUT_MARGIN_S):
    """Per-request HTTP budget for a sweep at this context depth.

    Replaces a flat `timeout=3600`, which on 2026-08-21 spent a full hour --
    01:34:36 to 02:34:36, to the second -- waiting on an arm whose prefill had
    already collapsed to 8.56 tok/s and could never have finished. A budget
    that does not know the depth is not a budget, it is a coincidence.
    """
    if ctx <= 0:
        raise ValueError(f"context must be positive, got {ctx}")
    prefill_tokens = ctx * PREFILL_FILL_FRACTION
    return prefill_tokens / floor_tok_s + margin_s


# A desktop compositor moves tens of MiB between polls. A 12 GB model unloading
# moves thousands. 64 separates them with two orders of magnitude to spare.
VRAM_SETTLE_TOL_MIB = 64


# The smallest artifact this project loads is 7.80 GiB. Any real teardown frees
# thousands of MiB, so demanding a tenth of that as proof of arrival is generous
# and still cannot be met by a release that has not started.
VRAM_MIN_RISE_MIB = 1024


def vram_settled(free_readings, tol_mib=VRAM_SETTLE_TOL_MIB, need=2,
                 floor_mib=None):
    """True once free VRAM has stopped moving AND cleared `floor_mib`.

    `kill()` used to sleep a flat 5 s. WDDM releases a 12 GB allocation in
    stages, so the next arm started into VRAM the driver still held, passed
    /health, and died on its first request with ConnectionResetError -- taking
    out a queue step that had nothing to do with the arm that was slow.

    One reading is never settled: that is the 5 s sleep restated.

    `floor_mib` exists because "stopped moving" alone is ambiguous between
    *release finished* and *release has not begun*: two polls taken before the
    driver does anything agree perfectly. The caller knows how much was free
    before the kill, so it can demand that the reading beat it -- which a
    release that never started cannot do. The check is against the LATEST
    reading, not the best one, so a transient spike mid-release is not arrival.
    """
    if len(free_readings) < max(2, need):
        return False
    window = free_readings[-need:]
    if max(window) - min(window) > tol_mib:
        return False
    return floor_mib is None or free_readings[-1] >= floor_mib


# The output contract every corpus arm is graded against. `check_output_contract`
# grades exactly this: one fenced python block, nothing else. It has been the
# whole developer message since the corpus was written.
CONTRACT = ("You are a precise Python engineer. Reply with one fenced ```python "
            "block containing only the requested code. No explanation, no usage "
            "examples, no tests.")


def compose_developer(skills):
    """Build the developer message from injected skill text plus the contract.

    Raised 2026-08-21: the real worker runs with `karpathy-guidelines` and `tdd`
    in its prompt and the corpus sends CONTRACT alone, so every quality number
    the project holds describes a configuration nobody ships.

    Two rules, both load-bearing.

    **The contract survives.** Replacing it would change two things at once --
    skills added AND the format instruction removed -- and an arm that changes
    two things cannot say which one moved the result. That is the fault that
    made the grammar/`-rea off` pair unreadable.

    **The contract goes last.** The injected skills contradict it directly: tdd
    says write the failing test first, the contract says no tests; karpathy says
    stop and ask when unclear, and a question is not a fenced block. Recency is
    the only lever available for deciding which instruction wins, so the graded
    requirement gets it.

    Skill text is passed through verbatim. A paraphrase would measure the
    paraphrase.
    """
    for s in skills:
        if not isinstance(s, str):
            raise TypeError(f"skill text must be str, got {type(s).__name__}")
    sep = chr(10) * 2
    return sep.join(list(skills) + [CONTRACT])


def line_repetition_pct(text):
    """Percentage of non-blank lines that are exact duplicates of an earlier line.

    Instrument fault 8, 2026-08-21. `depth_sweep.filler()` repeats one class
    definition with only a four-digit index changing -- 962 blocks at 147,456,
    adjacent blocks 99.5 % identical. An n-gram decoder drafts from what is
    already in the context, so that text is the most favourable input that could
    be constructed for it, and every n-gram figure this project holds was
    measured on it. Acceptance at 99-100 % across every depth is the tell.

    This makes the property checkable. A filler intended to stand in for real
    code should score low; the current one scores high, and the difference is
    the size of the correction owed to the headline numbers.

    Blank lines are excluded: they repeat in any text and would inflate every
    score toward 100 % regardless of content.
    """
    lines = [l for l in text.splitlines() if l.strip()]
    if not lines:
        raise ValueError("cannot measure repetition of empty text")
    seen = set()
    repeats = 0
    for l in lines:
        if l in seen:
            repeats += 1
        seen.add(l)
    return round(100.0 * repeats / len(lines), 2)


# Written to check a benchmark prompt; its more valuable use turned out to be a
# reasoning trace. Same function, the name the caller needs.
filler_repetition_pct = line_repetition_pct


def window_repetition_pct(text, n=24):
    """Percentage of n-word windows that have already appeared earlier in the text.

    The repetition measure that matches what `ngram-mod` actually sees.
    `line_repetition_pct` above answers "how much of this is duplicated lines",
    which is the right question for the tiled filler that produced instrument
    fault 8. It is the wrong question for choosing a corpus, because ngram-mod
    keys a hash on a window of n_match TOKENS (common/ngram-mod.cpp:15-25) and
    never looks at a line.

    The two disagree in both directions, and each disagreement matters:

      - 43 files of this repo's own source score 18.9 % on lines, almost
        entirely from `import sys`, `try:` and docstring delimiters. Those are
        not n-gram hits: the 24 tokens surrounding each occurrence differ. Real
        multi-file code always looks like this, and rejecting a corpus for it
        would be rejecting honest text.
      - Text tiled with a changing index -- exactly what `filler()` did -- can
        have every line differ while its windows repeat almost perfectly. That
        is the case that manufactures a fake n-gram verdict, and the line
        metric can miss it.

    PROXY, DELIBERATELY. Windows are n whitespace-separated words, not n llama
    tokens: the tokenizer is not available here and loading it to score a corpus
    would tie the measure to an artifact. Use this to rank candidate corpora
    against each other and against a known-bad filler. It does not predict an
    absolute hit rate and must not be quoted as one.
    """
    if n <= 0:
        raise ValueError("window width must be positive, got %r" % (n,))
    words = text.split()
    if len(words) < n:
        return 0.0
    seen = set()
    repeats = 0
    total = len(words) - n + 1
    for i in range(total):
        w = tuple(words[i:i + n])
        if w in seen:
            repeats += 1
        seen.add(w)
    return round(100.0 * repeats / total, 2)


def draft_acceptance(timings):
    """Percentage of drafted tokens accepted, over EVERY timed generation.

    Instrument fault 9, 2026-08-21. `depth_sweep.run()` reports `tg_med` as the
    median of five generations and computed `acceptance` from the first one
    alone, so the two columns described different requests. An arm could show
    "acceptance: null" -- meaning the cold request drafted nothing -- while its
    four warm requests drafted and were accepted, and its decode rate said so.

    Weighted by drafts, not averaged over requests: a generation that drafted
    nothing has no opinion about the acceptance rate and must not pull it toward
    zero. Returns None when nothing was drafted anywhere, which is a different
    fact from 0 % and has to stay distinguishable.
    """
    drafted = accepted = 0
    for t in timings:
        d = int(t.get("draft_n") or 0)
        a = int(t.get("draft_n_accepted") or 0)
        if a > d:
            raise ValueError(f"accepted {a} > drafted {d}: not a measurement")
        drafted += d
        accepted += a
    if not drafted:
        return None
    return round(100.0 * accepted / drafted, 1)


def cache_reuse_pct(timings):
    """Share of the submitted prompt that llama-server served from cache.

    `cache_n` + `prompt_n` is the whole prompt: tokens reused, plus tokens
    actually evaluated this request. The ratio is what says whether an agent
    turn re-prefilled or appended.

    Reads both fields with `[]`, deliberately. The 2026-08-22 prefix-cache run
    used `.get()` with a default, so a response whose timings block lacked
    `cache_n` -- which is what /v1/chat/completions returns, against the raw
    /completion endpoint this needs -- would have rendered as a 0 % row and read
    as a real cache miss. Pointing a driver at the wrong endpoint has to be a
    crash, not a page of plausible zeros.

    Both counters zero is not 0 % either: nothing was submitted, so the question
    was never asked.
    """
    cached = timings["cache_n"]
    evaluated = timings["prompt_n"]
    if cached is None or evaluated is None:
        raise TypeError("cache_n/prompt_n present but null: not a measurement")
    if cached < 0 or evaluated < 0:
        raise ValueError(f"negative token counter: cache_n={cached} prompt_n={evaluated}")
    total = cached + evaluated
    if total == 0:
        raise ValueError("empty prompt: reuse percentage is undefined, not 0 %")
    return 100.0 * cached / total


NOISE_FLOOR_PCT = 13.6   # measured restart-to-restart peak-to-peak; report 04 s0


def paired_deltas(baseline_rounds, candidate_rounds, floor_pct=NOISE_FLOOR_PCT):
    """Per-round paired differences between two arms measured in alternating boots.

    Two different models cannot share a boot, so the arms are alternated
    (A/B/A/B) and paired by round. Pairing is what keeps the 13.6 % restart
    drift inside each round instead of letting it land on whichever arm ran
    later -- the exact failure that manufactured a "+11.6 %" speculative result
    which reversed to -0.8 % against a fresh control.

    Returns per-round percentages AND a range, never a single point, and marks
    an effect `resolved` only when it is both larger than the drift floor and
    consistent in sign across rounds. A mean of +40/-10 is not an effect.
    """
    if not baseline_rounds or not candidate_rounds:
        raise ValueError("paired_deltas needs at least one round in each arm")
    if len(baseline_rounds) != len(candidate_rounds):
        raise ValueError(
            f"arms are not paired: {len(baseline_rounds)} baseline rounds vs "
            f"{len(candidate_rounds)} candidate rounds"
        )

    per_round = []
    for i, (b, c) in enumerate(zip(baseline_rounds, candidate_rounds), 1):
        if b <= 0:
            raise ValueError(f"round {i} baseline must be positive, got {b}")
        per_round.append(round(100.0 * (c - b) / b, 2))

    mean_pct = round(sum(per_round) / len(per_round), 2)
    same_sign = all(d > 0 for d in per_round) or all(d < 0 for d in per_round)
    resolved = bool(
        len(per_round) >= 2 and same_sign and abs(mean_pct) >= floor_pct
    )
    return {
        "rounds": len(per_round),
        "per_round_pct": per_round,
        "mean_pct": mean_pct,
        "min_pct": min(per_round),
        "max_pct": max(per_round),
        "consistent_sign": same_sign,
        "resolved": resolved,
        "floor_pct": floor_pct,
    }


def check_tool_call(tool_calls, spec):
    """Score one assistant turn against an expected tool-call contract.

    Returns every fault, not the first: the research asks for a required-field
    OMISSION RATE and a MALFORMED-CALL RATE, and a boolean cannot distinguish
    one dropped field from a model that answered in prose. Prose instead of a
    call is the degradation this exists to catch -- the code corpus alone would
    have scored it identically to a perfect call.
    """
    import json

    errors = []
    if not tool_calls:
        return {"ok": False, "errors": ["no tool call emitted"], "args": None}

    fn = (tool_calls[0] or {}).get("function") or {}
    name = fn.get("name")
    if name != spec["name"]:
        errors.append("wrong function name %r, expected %r" % (name, spec["name"]))

    raw = fn.get("arguments")
    args = None
    if isinstance(raw, dict):
        args = raw
    else:
        try:
            args = json.loads(raw or "")
        except (json.JSONDecodeError, TypeError) as e:
            errors.append("arguments are not valid JSON: %s" % e)

    if isinstance(args, dict):
        for field in spec.get("required", []):
            if field not in args:
                errors.append("required field %r missing" % field)
    elif args is not None:
        errors.append("arguments decoded to %s, expected an object" % type(args).__name__)

    if len(tool_calls) > spec.get("max_calls", 1):
        errors.append("%d calls emitted, expected at most %d"
                      % (len(tool_calls), spec.get("max_calls", 1)))

    return {"ok": not errors, "errors": errors, "args": args}


def retry_economics(records, escalation_s, overhead_s):
    """Turn per-task retry records into the units the decision is actually made in.

    Each record is {"attempts": int, "accepted": bool, "wall_s": float}, where
    `accepted` means the LOCAL worker eventually produced a passing patch. A task
    the worker never got right is not lost -- it escalates to Q4 -- so every task
    is merged, and escalation is charged as time rather than as a failure.

    p2 is None, never 0, when nothing needed a retry: a rate over an empty
    denominator reads as "retries never work" instead of "no retry happened".
    """
    if not records:
        raise ValueError("retry_economics needs at least one task record")

    n = len(records)
    first_pass = retried = retried_ok = accepted = attempts_total = 0
    censored = 0
    wall = 0.0
    for i, r in enumerate(records, 1):
        a = int(r["attempts"])
        ok = bool(r["accepted"])
        if a < 1:
            raise ValueError("record %d has %d attempts; a task cannot be "
                             "accepted or rejected without being attempted" % (i, a))
        attempts_total += a
        wall += float(r["wall_s"])
        # A task whose attempt ended at the token limit is CENSORED, not failed:
        # the model was still writing. Counting it as a failure penalises the
        # artifacts that reason longest, which is the same bias an undersized
        # budget produces, one notch quieter -- at max_tokens 8192 the arms
        # still truncate 1 to 7 times out of 60.
        if r.get("censored"):
            censored += 1
            continue
        if ok:
            accepted += 1
            if a == 1:
                first_pass += 1
        if a >= 2:
            retried += 1
            if ok:
                retried_ok += 1

    request_failed = sum(1 for r in records if r.get("request_failed"))
    # A run whose requests mostly never reached the model is not a slow run.
    # The earlier guard only fired when NO task recorded worker time, so a run
    # that completed four tasks and then lost its server to a colliding queue
    # still produced an ordinary-looking summary. 10 % is generous; anything
    # above it means the machine, not the model, decided the outcome.
    if n and request_failed / n > 0.10:
        raise ValueError(
            "%d of %d tasks failed before reaching the model (%.0f %%). The "
            "server was unavailable for most of this run, so nothing here "
            "describes the artifact." % (request_failed, n, 100.0 * request_failed / n))

    decided = n - censored
    if decided <= 0:
        raise ValueError(
            "all %d tasks were censored by the token budget; raise it and re-run "
            "rather than reading a pass rate off nothing" % n)

    if wall <= 0:
        raise ValueError(
            "no worker time recorded across %d tasks -- every request failed "
            "before the model ran. Escalation and overhead are constants, so a "
            "summary built from this would still print a plausible "
            "merged_tasks_per_hour." % n)

    escalations = decided - accepted
    total_s = wall + escalation_s * escalations + overhead_s * n

    # CAPABILITY and THROUGHPUT are reported apart, because summing them into
    # one number is how four arms that all scored 27/30 came to look like a
    # ranking. They differed only in wall clock -- 2,004 s to 4,572 s -- which
    # is verbosity, not skill. `accepted_of_decided` is the capability axis;
    # `wall_per_accepted_s` is the throughput axis; `merged_tasks_per_hour`
    # remains for continuity and is the two of them multiplied together.
    return {
        "tasks": n,
        "decided": decided,
        "request_failures": request_failed,
        "censored": censored,
        # A verdict that one censored task could flip is not a verdict. Compare
        # the best and worst case the censored tasks could produce.
        "censoring_could_change_verdict": bool(
            censored and accepted != accepted + censored),
        "p1": round(100.0 * first_pass / decided, 1),
        "p2": round(100.0 * retried_ok / retried, 1) if retried else None,
        "accepted_of_decided": "%d/%d" % (accepted, decided),
        "local_accept_pct": round(100.0 * accepted / decided, 1),
        "attempts_per_accepted": round(attempts_total / accepted, 2) if accepted else None,
        "escalations_per_100": round(100.0 * escalations / decided, 1),
        "worker_wall_s": round(wall, 1),
        "wall_per_accepted_s": round(wall / accepted, 1) if accepted else None,
        "escalation_s": escalation_s,
        "overhead_s": overhead_s,
        "merged_tasks_per_hour": round(3600.0 * n / total_s, 1) if total_s else None,
        "verified_tasks_per_hour": round(3600.0 * accepted / wall, 1) if wall else None,
    }


def marginal_rate(xs, ys, project_to=None):
    """Least-squares slope of y against x, reported as a RATE (x units per y unit).

    Written for the prefix-invalidation curve: y is the wall time of a turn
    whose cache was thrown away, x is the tokens it had to re-evaluate. Wall
    time also contains a decode of n_predict tokens, but that component is the
    same at every point, so it lands in the intercept and leaves the slope
    clean. Dividing a single point instead -- which is what this project did
    first -- charges the whole decode to the prefill and understates the rate.

    Three points minimum: two always fit a line exactly and report a perfect
    residual, which reads as certainty rather than as having no evidence.
    """
    if len(xs) != len(ys):
        raise ValueError("marginal_rate needs paired xs and ys, got %d and %d"
                         % (len(xs), len(ys)))
    n = len(xs)
    if n < 3:
        raise ValueError("marginal_rate needs at least 3 points, got %d" % n)

    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        raise ValueError("all x values are identical; no slope is defined")
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    if slope <= 0:
        raise ValueError("slope is %.6g; y must increase with x for a rate" % slope)
    offset = my - slope * mx

    syy = sum((y - my) ** 2 for y in ys)
    r2 = 1.0 if syy == 0 else 1.0 - sum(
        (y - (slope * x + offset)) ** 2 for x, y in zip(xs, ys)) / syy

    out = {
        "points": n,
        "rate": round(1.0 / slope, 1),     # x units per unit of y
        "offset_s": round(offset, 2),
        "r2": round(r2, 4),
    }
    if project_to is not None:
        out["project_to"] = project_to
        out["projected_s"] = round(project_to * slope + offset, 1)
    return out


def check_output_contract(text):
    """Did the reply obey the corpus's stated output format, before extraction?

    The corpus prompt asks for one fenced ```python block containing only the
    requested code, with no explanation, usage example or tests. The extractor
    then takes the LARGEST fenced block, or the whole reply when there is no
    fence -- which quietly repairs every violation and lets a disobedient reply
    score exactly like an obedient one.

    A review panel identified this as the capability aggressive quantization is
    said to lose FIRST, ahead of closed algorithmic coding, and the corpus was
    structurally unable to see it. This measures it as a separate rate. It is
    deliberately NOT wired into pass/fail: redefining a passing task mid-project
    would make every number collected before today incomparable.
    """
    import re

    violations = []
    blocks = re.findall(r"```[ \t]*(?:python|py)?[ \t]*\n(.*?)```", text or "", re.S)
    if not blocks:
        violations.append("no fenced python block")
        return {"ok": False, "violations": violations, "blocks": 0}
    if len(blocks) > 1:
        violations.append("%d fenced blocks, expected 1" % len(blocks))

    outside = re.sub(r"```[ \t]*(?:python|py)?[ \t]*\n.*?```", "", text, flags=re.S)
    if outside.strip():
        violations.append("prose outside the fence: %r"
                          % outside.strip()[:60])

    body = "\n".join(blocks)
    if re.search(r"^\s*if\s+__name__\s*==", body, re.M):
        violations.append("usage example inside the block (__main__ guard)")

    return {"ok": not violations, "violations": violations, "blocks": len(blocks)}


_SPEC_IMPL_RE = None


def parse_spec_impl_stats(log_text):
    """Per-implementation speculation counters, keyed by implementation name.

    With a chained `--spec-type draft-dflash,ngram-mod` the summary line pools
    both speculators:

        draft acceptance = 0.46013 (352 accepted / 765 generated), mean len = 3.26

    and the pooled number hides the finding. From the run behind report 29:

        ngram-mod    : #calls(b,g,a) = 4, 542,  31, mean acc len = 18.00
        draft-dflash : #calls(b,g,a) = 4, 511, 511, mean acc len =  2.91

    `ngram-mod` was asked 542 times and produced a draft 31 times -- it declines
    94.3 % of the time -- and `draft-dflash` was called exactly the 511 times
    ngram declined. When ngram does fire it is worth six times more per draft.

    The declines are `common/speculative.cpp:1993`: when the n-gram table misses
    before `n_min` successors the whole draft is discarded, not truncated. So
    `--spec-ngram-mod-n-min` is a fire-rate knob and `decline_pct` is how its
    effect is read.

    THE COUNTERS ARE CUMULATIVE and the server reprints them after every
    completion, so the LAST block is the run and the first block is the first
    task. Returns {} for a log that has no such lines at all -- absent is a
    different fact from zero, and a log written below LOG_TRC has none.
    """
    global _SPEC_IMPL_RE
    if _SPEC_IMPL_RE is None:
        import re
        _SPEC_IMPL_RE = re.compile(
            r"statistics\s+(?P<name>[\w.-]+):\s*"
            r"#calls\(b,g,a\)\s*=\s*(?P<cb>\d+)\s+(?P<cg>\d+)\s+(?P<ca>\d+),\s*"
            r"#gen drafts\s*=\s*(?P<gd>\d+),\s*"
            r"#acc drafts\s*=\s*(?P<ad>\d+),\s*"
            r"#gen tokens\s*=\s*(?P<gt>\d+),\s*"
            r"#acc tokens\s*=\s*(?P<at>\d+)"
            r"(?:,\s*#mean acc len\s*=\s*(?P<mal>[\d.]+))?"
            r"(?:.*?dur\(b,g,a\)\s*=\s*(?P<tb>[\d.]+),\s*(?P<td>[\d.]+),\s*(?P<ta>[\d.]+)\s*ms)?"
        )

    out = {}
    for m in _SPEC_IMPL_RE.finditer(log_text):
        g = m.groupdict()
        n_call_draft = int(g["cg"])
        n_gen_drafts = int(g["gd"])
        # None, not 0.0: an implementation that was never asked has no decline
        # rate, and reporting 0 % would read as "it always fired".
        decline = (round(100.0 * (n_call_draft - n_gen_drafts) / n_call_draft, 1)
                   if n_call_draft else None)
        out[g["name"]] = {
            "n_call_begin":  int(g["cb"]),
            "n_call_draft":  n_call_draft,
            "n_call_accept": int(g["ca"]),
            "n_gen_drafts":  n_gen_drafts,
            "n_acc_drafts":  int(g["ad"]),
            "n_gen_tokens":  int(g["gt"]),
            "n_acc_tokens":  int(g["at"]),
            "mean_acc_len":  float(g["mal"]) if g["mal"] else None,
            "t_begin_ms":    float(g["tb"]) if g["tb"] else None,
            "t_draft_ms":    float(g["td"]) if g["td"] else None,
            "t_accept_ms":   float(g["ta"]) if g["ta"] else None,
            "decline_pct":   decline,
        }
    return out


# A generation that produced almost nothing has not measured a decode rate. A
# quarter of the requested budget is generous -- a model that answers in 300 of
# 512 tokens has done real work; one that answers in 4 has not.
MEASURABLE_FRACTION = 0.25


def generation_is_measurable(timings, n_predict, fraction=MEASURABLE_FRACTION):
    """True when every timed generation in a row actually generated.

    INSTRUMENT FAULT, 2026-08-22. The draft-count sweep re-run at ctx 65,536
    reported a tight, RESOLVED -56.5 % for the widest arm, with every arm
    showing acceptance 0.0 and `mean len 1.0` -- which read as "speculation
    stops working at depth". The server log said:

        eval time = 112.32 ms /     4 tokens
        eval time =  61.45 ms /     2 tokens

    The generations produced two to four tokens against a 512-token budget: the
    frozen corpus is ~28,000 tokens and the arena had asked for a 32,768-token
    prompt, so the whole corpus was consumed and the model answered in a few
    tokens. Three arms, six rows, a tight range and a resolved verdict, all
    computed over noise.

    The previous guard refused a rate of zero and let a rate over four tokens
    through, so the failure arrived as a plausible number rather than as an
    error. That is the one outcome this project treats as worse than a crash.

    ALL samples must qualify, not the median: a row is one paired datapoint, and
    a median over a good sample and a 3-token sample is not a measurement of
    anything. That is how it got through the first time.
    """
    if not timings:
        return False
    floor = max(1, int(n_predict * fraction))
    for t in timings:
        n = t.get("predicted_n") or 0
        r = t.get("predicted_per_second") or 0
        if n < floor or r <= 0:
            return False
    return True


# Live project checkouts. The real-task benchmark reads issues from these and
# must never write to or delete anything under them: on 2026-08-22 MangaDock
# had 333 uncommitted files on a feature branch and T4 Fastwork 440 plus four
# stashes -- days of work existing nowhere else. The benchmark ends by deleting
# what it made, so an unguarded path turns cleanup into destruction.
PROTECTED_ROOTS = (
    r"D:\Github",
)


def is_protected(path):
    """True if `path` is a protected root or lives under one.

    Case-insensitive, because Windows is, and a guard that `D:\\github\\x`
    slips past is not a guard.
    """
    import os
    p = os.path.normcase(os.path.abspath(str(path)))
    for root in PROTECTED_ROOTS:
        r = os.path.normcase(os.path.abspath(root))
        if p == r or p.startswith(r + os.sep):
            return True
    return False


def assert_deletable(path, scratch_root):
    """Raise unless `path` is inside `scratch_root` and nothing is protected.

    Checked BEFORE any delete, not after. Three ways to fail, each a real one:

    - the path is a live checkout, or inside one;
    - the path is outside the run's declared scratch root, so deleting it was
      not something this run planned;
    - the scratch root is itself protected, which would let every other check
      pass while guarding nothing.

    A relative path is refused rather than resolved, because what it names
    depends on the working directory and that is not something a benchmark
    controls.
    """
    import os
    if not os.path.isabs(str(path)) or not os.path.isabs(str(scratch_root)):
        raise ValueError("refusing a relative path: %r (root %r)" % (str(path), str(scratch_root)))
    if is_protected(scratch_root):
        raise ValueError("scratch root is protected: %r" % str(scratch_root))
    if is_protected(path):
        raise ValueError("refusing to delete a protected path: %r" % str(path))
    p = os.path.normcase(os.path.abspath(str(path)))
    r = os.path.normcase(os.path.abspath(str(scratch_root)))
    if not (p == r or p.startswith(r + os.sep)):
        raise ValueError("refusing to delete outside the scratch root: %r not under %r"
                         % (str(path), str(scratch_root)))
    return True


# Within 2 % of the window counts as saturated. Proportional rather than a
# fixed token count, because a 4,096-token window is full long before 32,767.
WINDOW_SATURATION = 0.98


def classify_outcome(verify_exit, changed_files, ctx_high_water, n_ctx,
                     saturation=WINDOW_SATURATION):
    """PASS / FAIL / WINDOW_BOUND for one real-task attempt.

    WINDOW_BOUND exists because of a run on 2026-08-22 that reported

        5 tasks: 0 PASS, 5 FAIL, 0 VOID
        context high-water: min 32767  median 32767  max 41377

    against a 32,768-token window, with every baseline green. It reads as a
    verdict on the worker -- nought for five -- and it is not one. 32,767 is
    `n_ctx - 1`, and the server log carried `exceeds the available context size
    (32768 tokens)` six times. The tasks filled the window; the model was never
    given room to finish.

    Counting that as FAIL blames the model for a number the operator chose.

    WINDOW_BOUND is not a worker failure and must never be totalled as one. It
    is still a RESULT: it says this class of task does not fit that window,
    which is the whole of the context-sizing question.

    A PASS that also saturated stays a PASS -- it got there, and running close
    to the edge is not a disqualification. An unknown high-water stays FAIL:
    missing data is not evidence of a missing window, and excusing a failure on
    absent evidence is how a benchmark stops reporting failures at all.
    """
    passed = (verify_exit == 0 and changed_files > 0)
    if passed:
        return "PASS"
    if ctx_high_water is not None and n_ctx and ctx_high_water >= n_ctx * saturation:
        return "WINDOW_BOUND"
    return "FAIL"
