# -*- coding: utf-8 -*-
"""Which llama-server ran, and what GPU architectures it was compiled for.

WHY THIS IS A MODULE AND NOT TWO LINES IN A SCRIPT.

On 2026-08-24 this machine held three llama-server delivery directories. Two of
them -- `llama.cpp-cuda` and `llama.cpp-dflash2` -- carry SASS for `sm_89` only,
while the installed card is `sm_120`. The third, `llama.cpp-blackwell`, is the
same tree at the same commit built with `CMAKE_CUDA_ARCHITECTURES="89;120"`.

They are indistinguishable by every ordinary means:

    version: 0.1.2-dev (build 10499, commit 1deefcca3)
    built with MSVC 19.44.35228.0 for Windows AMD64

That string is identical across all of them, as are the buffer sizes, the layer
split and the `--fit` decision -- while prefill differs by 2.20x, because the
driver silently JIT-compiles Ada PTX for the ones that lack Blackwell code
objects (docs/results/09-hardware.md).

So provenance cannot come from the binary's self-report. It has to come from the
code objects, and every script that launches a server needs the same answer, or
the next JSONL is full of rates nobody can attribute.

WHY resolve_exe TAKES A DEFAULT. Six scripts in this folder each named their own
binary. Rewriting those constants would have made the older figures
unreproducible and would not stop the seventh script from hardcoding a path
again. Taking the default as an argument leaves each script's behaviour
byte-identical while giving the operator one lever and the recorder one function.
`tests/test_every_script_routes_its_exe.py` holds that property.
"""
import os
import re
import subprocess
from pathlib import Path

ENV_VAR = "QWEN38_LLAMA_EXE"

# Separate from ENV_VAR on purpose. One variable for both would make pointing at
# a different model silently also point at a different binary, which is the
# confound this module exists to prevent rather than create.
TARGET_ENV_VAR = "QWEN38_TARGET"

# Shipped with the CUDA toolkit; the only tool here that reads code objects.
CUOBJDUMP = (r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3"
             r"\bin\cuobjdump.exe")

# cuobjdump on the 84 MB Blackwell dll takes seconds, and run_arm() asks once
# per row. The answer cannot change while the process runs.
_ARCH_CACHE = {}


def _from_env(var, default):
    """`var` if it is set to something non-empty, else `default`.

    The `or` is deliberate: an empty variable is a plausible operator slip
    (`$env:X = ""`), and treating it as "set" would launch the empty string and
    fail somewhere far from the cause.
    """
    return os.environ.get(var) or default


def resolve_exe(default):
    """The server binary to launch: `QWEN38_LLAMA_EXE` if set, else `default`."""
    return _from_env(ENV_VAR, default)


def resolve_target(default):
    """The model file to load: `QWEN38_TARGET` if set, else `default`.

    Added for the same reason as `resolve_exe`, one question later. Comparing
    two quantisations of one model means running the identical arm on two files;
    editing the module constant to do it makes the first figure unreproducible,
    and `docs/results/01-artifacts.md` lists TWO different files both named
    `Qwen3.8-27B-UD-Q2_K_XL.gguf` in different snapshot directories, so a
    filename cannot identify the artifact either.
    """
    return _from_env(TARGET_ENV_VAR, default)


def model_size_mib(path):
    """Size of `path` in MiB, or None if it is not there.

    Recorded beside the path because the path alone is not an identity: the two
    `UD-Q2_K_XL` files on this machine differ by 808 MiB, and a moved or
    re-synced cache would make a recorded path point at something else without
    changing a single character of the row.

    None rather than 0.0 -- an empty model and a missing one are different
    failures, and the missing one is the case where the row is already wrong.
    """
    try:
        return os.path.getsize(path) / 1048576
    except OSError:
        return None


def cuda_archs(exe):
    """Architectures compiled into the `ggml-cuda.dll` beside `exe`.

    Returns a sorted tuple such as ``("sm_120a", "sm_89")``, or **None** when it
    could not be determined -- missing dll, missing cuobjdump, non-zero exit.

    None means "not checked", never "none present". Conflating those is how an
    unverified build gets recorded as a verified one, so callers that care must
    branch on `is None` before branching on membership.

    The pattern is ``sm_\\w+`` rather than ``sm_\\d+``: llama.cpp's cmake rewrites
    `120` to `120a` and names the cubins `sm_120a`. Matching only digits reports
    a Blackwell binary as "sm_120", dropping the suffix that says which
    instruction set it actually carries.
    """
    if exe in _ARCH_CACHE:
        return _ARCH_CACHE[exe]
    dll = Path(exe).parent / "ggml-cuda.dll"
    archs = None
    if dll.is_file() and Path(CUOBJDUMP).is_file():
        try:
            out = subprocess.run([CUOBJDUMP, "--list-elf", str(dll)],
                                 capture_output=True, text=True, timeout=300)
            if out.returncode == 0:
                archs = tuple(sorted(set(re.findall(r"sm_\w+", out.stdout))))
        except (OSError, subprocess.SubprocessError):
            pass  # archs stays None: could not check, which is not "none found"
    _ARCH_CACHE[exe] = archs
    return archs
