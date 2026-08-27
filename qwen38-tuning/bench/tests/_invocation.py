"""Where a worker profile's argv lives, for tests that must read the VALUE.

Six test files sliced the source from the literal marker `& $Exe -m $Model` and
regexed the text after it. That was already the best available reading -- a
slice scoped to the invocation cannot be fooled by a flag mentioned in a
comment, which is the failure `docs/agents/traps.md` 16 is about.

It stopped working on 2026-08-27, when `worker-q4-dual.ps1` was rebuilt to
assemble `$argv` as an ARRAY so that `-WhatIf` could print exactly what would
run. Twelve assertions went red while the profile served the identical command
line. A test calling a refactor a regression is the same defect one level up.

PREFER THE DRY RUN. A profile with `-WhatIf` resolves the split, the window and
every argument and then exits without launching, so what it prints is the value
itself rather than a rendering of the source. Fall back to the source slice for
profiles that have no dry run yet.

NOTE WHAT THE FALLBACK CANNOT SEE: a value computed at runtime. `-c` in the dual
profile is rewritten by `-MaxCtx` and by `-Dflash`, and the source slice shows
`"$Ctx"`. Only the dry run resolves it.
"""
import os
import re
import subprocess

# Assembled-as-array form first, then the direct-call form.
_MARKERS = ("$argv = @(", "& $Exe -m $Model")


def _has_dry_run(path):
    with open(path, encoding="utf-8", errors="replace") as fh:
        return "WhatIfPreference" in fh.read()


def from_source(path):
    """The invocation region of the file, scoped so prose cannot be mistaken
    for a flag."""
    with open(path, encoding="utf-8", errors="replace") as fh:
        t = fh.read()
    for m in _MARKERS:
        if m in t:
            return t[t.index(m):]
    raise AssertionError(
        "no invocation found in %s -- looked for %r. If the launch was "
        "restructured again, teach this helper the new shape rather than "
        "loosening the assertions that use it." % (path, _MARKERS))


def resolved(path, *args, timeout=180):
    """What the profile would actually run, via its own -WhatIf. Falls back to
    the source slice when the profile has no dry run."""
    if not _has_dry_run(path):
        return from_source(path)
    r = subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
         "-File", path, "-WhatIf"] + list(args),
        capture_output=True, text=True, timeout=timeout)
    out = r.stdout + r.stderr
    assert "WhatIf: would run" in out, (
        "%s has -WhatIf but did not preview: %s" % (os.path.basename(path), out))
    return out


def flag(text, name):
    """The value following `name`, tolerating the quoted array form."""
    m = re.search(re.escape(name) + r"['\"]?\s*,?\s*['\"]?([^\s',\"]+)", text)
    return m.group(1) if m else None
