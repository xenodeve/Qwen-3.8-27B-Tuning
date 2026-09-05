r"""The EXL3 launchers' three knobs must reach serve-exl3.cmd intact THROUGH
REAL cmd.exe (2026-09-04, hub key G).

THE INCIDENT. `call serve-exl3.cmd 262144 10,15.5 127.0.0.1` looked like three
arguments and was four: cmd treats a comma as an argument separator, so the
recipe saw `-gs 10` (one card, the 4070 alone), the host became `15.5`, and
the model could not load. test_exl3_launcher.py read the line as text and
passed. This test hands the launcher's exact argument string to a stub .cmd
that reports what %~1 %~2 %~3 actually are.
"""
import os
import re
import subprocess
import sys
import tempfile

import pytest

BENCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(BENCH))
LAUNCHERS = os.path.join(ROOT, "launchers")
EXPECT = {
    "serve-exl3.bat": ("163840", "9,15.5", "127.0.0.1"),
    "serve-exl3-lan.bat": ("163840", "9,15.5", "0.0.0.0"),
    "serve-exl3-max.bat": ("262144", "9,15.5", "127.0.0.1"),
    "serve-exl3-max-lan.bat": ("262144", "9,15.5", "0.0.0.0"),
}
STUB = "@echo off\r\necho [%~1] [%~2] [%~3] [%~4]\r\n"


def argument_string(name):
    with open(os.path.join(LAUNCHERS, name), "rb") as fh:
        body = fh.read().decode("ascii")
    lines = [l for l in body.splitlines() if "serve-exl3.cmd\"" in l and l.strip().lower().startswith("call ")]
    assert len(lines) == 1, lines
    return lines[0].split('serve-exl3.cmd"', 1)[1].strip()


@pytest.mark.skipif(sys.platform != "win32", reason="needs the real cmd.exe parser")
@pytest.mark.parametrize("name,expect", EXPECT.items())
def test_the_three_knobs_arrive_as_three_arguments(name, expect):
    args = argument_string(name)
    with tempfile.TemporaryDirectory() as d:
        stub = os.path.join(d, "stub.cmd")
        with open(stub, "wb") as fh:
            fh.write(STUB.encode("ascii"))
        out = subprocess.run('cmd /c ""%s" %s"' % (stub, args), shell=True,
                             capture_output=True, text=True, timeout=30).stdout.strip()
    got = re.findall(r"\[([^\]]*)\]", out)
    assert got == [expect[0], expect[1], expect[2], ""], (name, args, out)
