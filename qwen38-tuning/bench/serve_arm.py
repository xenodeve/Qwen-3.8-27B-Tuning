# -*- coding: utf-8 -*-
"""Boot one server for a NAMED ARM from the arena, and hold it.

WHY THE ARM COMES FROM `ARM_SETS` AND IS NOT RETYPED HERE.

The point of running a real task on `dflash2+ngram` is to ask whether the arm
that won the throughput sweep also wins on the metric that counts -- verified
accepted coding tasks. That question is only meaningful if the server the task
talks to is the server the sweep measured. Retyping the flags into this file
would give a second definition of the arm, free to drift from the first, and the
drift would be invisible: both would boot, both would serve, and the task result
would silently belong to a configuration nothing had measured.

So the argv is looked up by label. If the arm is renamed or its flags change,
this script fails loudly instead of serving something else.

`real_task_bench.py` deliberately does not start a server -- "starting the thing
under test from inside the measurement is how you end up measuring the starter"
-- so this is the separate step that satisfies it.

    python serve_arm.py dflash2+ngram --ctx 98304
"""
import argparse
import sys
import time

import dflash2_arena as A


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arm", help="a label from dflash2_arena.ARM_SETS")
    ap.add_argument("--ctx", type=int, required=True)
    ap.add_argument("--arms", default="decoders",
                    help="which arm set the label lives in")
    ap.add_argument("--boot-s", type=int, default=900)
    a = ap.parse_args()

    arms = A.ARM_SETS[a.arms]
    match = [x for x in arms if A.arm_parts(x)[0] == a.arm]
    if len(match) != 1:
        have = ", ".join(A.arm_parts(x)[0] for x in arms)
        print(f"no single arm {a.arm!r} in set {a.arms!r}; have: {have}")
        return 2
    label, extra, env = A.arm_parts(match[0])

    A.require_exclusive_port()

    print(f"arm    {label}")
    print(f"argv   {' '.join(A.server_argv(a.ctx, extra))}")
    print(f"env    {env or '{}'}")
    print(f"exe    {A.EXE}")
    print(f"archs  {A.cuda_archs(A.EXE)}")

    p, fh, log, free_before = A.start(a.ctx, extra, f"serve-{label.replace('+', '-')}",
                                      boot_s=a.boot_s, env=env)
    if p is None:
        print(f"BOOT FAILED -- see {log}")
        return 1

    print(f"UP     ctx={a.ctx}  free_before={free_before} MiB  free_now={A.vram()[1]} MiB")
    print(f"log    {log}")
    print("holding; stop with dflash2_arena.stop_server() or taskkill", flush=True)
    try:
        while p.poll() is None:
            time.sleep(5)
    except KeyboardInterrupt:
        pass
    print(f"server exited rc={p.returncode}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
