"""The one place in `bench/` that asks the driver about a GPU.

WHY THIS MODULE EXISTS (2026-08-26).

A second card was connected. Every VRAM reader in this repo had been written for
a machine with exactly one GPU, and `nvidia-smi --query-gpu=...` answers per
card -- so the day the second card appeared, seven modules here and four scripts
in `../scripts` all started reading something other than what they claimed.

Python failed loudly (`ValueError` out of `int()` on a two-line string) and
PowerShell failed silently, reporting the OTHER card's numbers with no error at
all. `CLAUDE.md` names the second one as the thing this project exists to
prevent, so the fix is not a better parse: it is that a reading must carry the
identity of the card it came from, or refuse to be a reading.

WHY UUID AND NOT INDEX.

`nvidia-smi -i 1` and `--main-gpu 1` both name a position in an enumeration.
Enumeration order is a function of the driver, the BIOS and which slots are
populated -- all of which change without anyone editing this file, and none of
which announce themselves. A UUID names the silicon. If the card is moved to
another slot, the index that used to mean "the 5060 Ti" quietly starts meaning
the other card, and every row recorded after that is mislabelled; with a UUID,
the same move produces `GpuNotPresent` and stops the sweep.

Verified 2026-08-26 that `nvidia-smi -i <uuid>` resolves and exits 6 with
"No devices were found" on an absent one, and that
`CUDA_VISIBLE_DEVICES=GPU-<uuid>` leaves llama-server seeing exactly one device.
"""
import os
import subprocess

# The RTX 5060 Ti 16 GB (sm_120). Every number in `docs/results/` from
# 2026-08-23 onward was measured on this card; pinning to it is what keeps the
# register comparable now that a second card shares the machine.
SERVED_GPU_UUID = "GPU-059b90e2-2b5c-00b8-f3ba-f6dea8de083e"
SERVED_GPU_NAME = "NVIDIA GeForce RTX 5060 Ti"


class GpuNotPresent(RuntimeError):
    """The named card is not installed.

    Raised rather than returning a default on purpose. A default here is a
    plausible number attributed to hardware that is not in the machine, which is
    worse than a stopped sweep.
    """


def _smi(args):
    return subprocess.run(["nvidia-smi"] + args,
                          capture_output=True, text=True)


def installed():
    """(uuid, name) for every card the driver can see. Used by error messages,
    so it must not itself raise when something is wrong."""
    r = _smi(["--query-gpu=uuid,name", "--format=csv,noheader"])
    out = []
    for line in (r.stdout or "").splitlines():
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",", 1)]
        if len(parts) == 2:
            out.append(tuple(parts))
    return out


def is_present(uuid=SERVED_GPU_UUID):
    return any(u == uuid for u, _ in installed())


def query(fields, uuid=SERVED_GPU_UUID):
    """Raw per-field strings for ONE card.

    `-i <uuid>` is what makes this single-valued. Without it nvidia-smi prints a
    line per card and any positional parse downstream is reading whichever card
    the driver happened to list first.
    """
    r = _smi(["-i", uuid, "--query-gpu=" + ",".join(fields),
              "--format=csv,noheader,nounits"])
    lines = [ln for ln in (r.stdout or "").splitlines() if ln.strip()]
    if r.returncode != 0 or not lines:
        raise GpuNotPresent(
            f"{uuid} is not installed (nvidia-smi exit {r.returncode}: "
            f"{(r.stderr or r.stdout or '').strip()!r}). "
            f"Installed: {installed()}")
    if len(lines) > 1:
        # Cannot happen with `-i` today. If it ever does, the ambiguity this
        # module exists to remove has come back, and silence would hide it.
        raise GpuNotPresent(
            f"{uuid} matched {len(lines)} cards, which makes the reading "
            f"ambiguous: {lines!r}")
    return [p.strip() for p in lines[0].split(",")]


def vram(uuid=SERVED_GPU_UUID):
    """(used_mib, free_mib) for ONE named card.

    LIMIT, and it matters for issue #51: when a model is split across two cards
    this reports one of them. Free VRAM on the 5060 Ti says nothing about
    whether layers spilled on the 4070, so a residency verdict built on this
    alone would be a verdict about half a model. Two-card arms must sum
    `vram(u) for u in devices` -- `total_vram()` below does it.
    """
    used, free = query(["memory.used", "memory.free"], uuid)
    return int(used), int(free)


def total_vram(uuids):
    """(used_mib, free_mib) summed over several cards.

    For a model split across devices. Free VRAM does not really add up -- a
    layer cannot straddle two cards -- so the sum is a CEILING on what will
    fit, not a promise. Residency still has to be read from the layer split in
    llama.cpp's own log, which is what `harness.parse_layer_split` is for.
    """
    used = free = 0
    for u in uuids:
        u_used, u_free = vram(u)
        used += u_used
        free += u_free
    return used, free


def free_vram(uuid=SERVED_GPU_UUID):
    return vram(uuid)[1]


def name(uuid=SERVED_GPU_UUID):
    return query(["name"], uuid)[0]


def index(uuid=SERVED_GPU_UUID):
    """Current enumeration index. For a human-readable log line ONLY -- never
    store it as the identity of a row, which is the mistake this module is
    about."""
    return int(query(["index"], uuid)[0])


def link(uuid=SERVED_GPU_UUID):
    """(pcie_gen, pcie_width) as the driver reports them RIGHT NOW.

    Both downtrain when the card is idle, so a reading taken between runs says
    nothing about the slot. Sample this while the GPU is working or do not quote
    it (issue #51, stage 4).
    """
    gen, width = query(["pcie.link.gen.current", "pcie.link.width.current"], uuid)
    return int(gen), int(width)


def visible_uuids():
    """The cards THIS PROCESS was pinned to, from `CUDA_VISIBLE_DEVICES`.

    `launch_env` covers a launcher that builds the child's environment. This
    covers the other shape: a script started with the pin already exported in
    the shell, which is how `ctx_ceiling.py` and the sweeps are run. Those have
    no arm-env to consult, and asking about the served card alone understates a
    two-card run -- the Q4 ladder on 2026-08-26 printed `free 4130` at ctx
    131,072 while the run had both cards' headroom.

    An index-style pin (`CUDA_VISIBLE_DEVICES=0,1`) is legal for CUDA and
    useless here, so it raises. Returning ["0", "1"] would send an index to
    `nvidia-smi -i`, which answers for a POSITION -- reintroducing exactly the
    ambiguity this module exists to remove, one layer further down.
    """
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not raw:
        return [SERVED_GPU_UUID]
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    bad = [p for p in parts if not p.startswith("GPU-")]
    if bad:
        raise GpuNotPresent(
            f"CUDA_VISIBLE_DEVICES names {bad!r} by index, not by UUID. An "
            f"index is a position in an enumeration and cannot identify a card "
            f"here. Installed: {installed()}")
    return parts


def visible_vram():
    """(used, free) summed over the cards this process can see."""
    return total_vram(visible_uuids())



def visible_compute_caps():
    """The compute capability of every GPU this process can see, in order.

    Returned as nvidia-smi gives them -- "8.9", "12.0" -- so they can be
    compared against the architectures llama.cpp prints in its `system_info`
    line (`harness.archs_missing_for_gpus`).

    Added 2026-08-27 after a sweep ran fifteen rows on a binary built with
    CMAKE_CUDA_ARCHITECTURES=89 while a capability 12.0 card was visible and in
    use. It lives here rather than at the call site because this module is the
    only place in the Python half allowed to ask the driver anything; a test
    forbids `--query-gpu` everywhere else.
    """
    return [query(["compute_cap"], u)[0] for u in visible_uuids()]

def pin_env(uuid=SERVED_GPU_UUID):
    """Environment that makes a child process see this card and no other.

    CUDA accepts a UUID here, so the pin survives a change of enumeration order
    the same way the readings above do. Checked first: an absent UUID leaves
    llama-server with zero devices, which it reports and then runs on CPU --
    plausible output, catastrophic rate, no flag in the row saying why.
    """
    if not is_present(uuid):
        raise GpuNotPresent(
            f"refusing to pin to {uuid}: not installed. Installed: {installed()}")
    return {"CUDA_VISIBLE_DEVICES": uuid}


def describe(uuid=SERVED_GPU_UUID):
    """One line for a log header, so a run says on its face which card it used.

    One `nvidia-smi` call, not four: composing this from `name`/`index`/`vram`/
    `link` reads the driver once per field, and the fields would then come from
    four different instants.
    """
    nm, idx, used, free, gen, width = query(
        ["name", "index", "memory.used", "memory.free",
         "pcie.link.gen.current", "pcie.link.width.current"], uuid)
    return (f"{nm} [{uuid}] index {idx} "
            f"used {int(used):,} MiB free {int(free):,} MiB "
            f"PCIe gen{gen} x{width}")
