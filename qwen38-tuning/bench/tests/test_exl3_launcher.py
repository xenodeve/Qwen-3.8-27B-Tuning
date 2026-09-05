r"""Hub keys F and G: the ExLlama3 fork server (issue #71, results 10).

WHY IT IS IN THE HUB. After six optimisation passes and one idle-boot pairing
the EXL3 recipe decodes at ~81 % of the served llama.cpp profile at 147K and,
in the developer's own sessions at 30-70K, 47-55 tok/s against llama.cpp's
39-47 (results 10; the 2026-09-02 serve logs). It is a second engine worth a
key, not a default: quality is unmeasured on any EXL3 arm, and it speaks the
OpenAI API only, so Claude Code reaches it through the LiteLLM proxy that
`claude-xeno-exl3` starts.

THE SAME RULE AS EVERY OTHER KEY: the hub holds no flags, the launcher holds no
flags, and the recipe is written in exactly one file --
qwen38-tuning\scripts\serve-exl3.cmd. A launcher that copied `-cq 4 -ndt 3`
would be the second source of truth this project has already shipped once.

TWO KEYS, NOT ONE. F serves the measured cache (163,840 tokens, the depth every
row in results 10 was taken at). G serves the model's native maximum (262,144).
Both keys use the split 9,15.5 since 2026-09-04: the served artifact became
turboderp's SC 4.00bpw H5 (VENDOR KL 0.0062 vs 3.5bpw unmeasured), which OOMs
the 4070 at cap 10 on a 197K prompt and passes at 9 (results 10).
"""
import os
import re

import pytest

BENCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(BENCH))
LAUNCHERS = os.path.join(ROOT, "launchers")
HUB = os.path.join(ROOT, "serve-hub.bat")
RECIPE = os.path.join(ROOT, "qwen38-tuning", "scripts", "serve-exl3.cmd")
PAIRS = {
    "serve-exl3.bat": "serve-exl3-lan.bat",
    "serve-exl3-max.bat": "serve-exl3-max-lan.bat",
}
BATS = sorted(list(PAIRS) + list(PAIRS.values()))
FLAGS = ("-cq", "-tp", "-tpb", "-gs", "-ndt", "-cs", "-dm", "--port")


def read(path):
    with open(path, "rb") as fh:
        return fh.read().decode("ascii")


def launch_line(text):
    lines = [l for l in text.splitlines()
             if l.strip().lower().startswith("call ") and "serve-exl3.cmd" in l]
    assert len(lines) == 1, lines
    return lines[0]


@pytest.mark.parametrize("name", BATS)
def test_the_launcher_exists_and_holds_no_serving_flag(name):
    path = os.path.join(LAUNCHERS, name)
    assert os.path.exists(path), path
    body = read(path)
    for line in body.splitlines():
        if line.strip().lower().startswith("rem"):
            continue
        for f in FLAGS:
            assert (" %s " % f) not in (line + " "), (name, f, line)
    assert "serve-exl3.cmd" in launch_line(body)


@pytest.mark.parametrize("loop,wide", PAIRS.items())
def test_the_lan_twin_differs_only_by_the_bind_address(loop, wide):
    a = launch_line(read(os.path.join(LAUNCHERS, loop)))
    b = launch_line(read(os.path.join(LAUNCHERS, wide)))
    assert "0.0.0.0" in b and "0.0.0.0" not in a
    assert b.replace("0.0.0.0", "127.0.0.1") == a, (a, b)


def test_max_passes_the_native_window_and_the_raised_cap():
    a = launch_line(read(os.path.join(LAUNCHERS, "serve-exl3.bat")))
    m = launch_line(read(os.path.join(LAUNCHERS, "serve-exl3-max.bat")))
    assert "163840" in a and "9,15.5" in a
    # 2026-09-04: G moved from 10,15.5 to 9,15.5 -- the SC 4.00bpw H5 file OOMs the 4070 at
    # 10 (sync timeouts after a 197K prefill) and passes the same probe at 9
    assert "262144" in m and "9,15.5" in m


def test_the_recipe_lives_in_one_file_and_is_the_measured_one():
    body = read(RECIPE)
    for tok in ("-cq 4", "-dm mtp", "-tp -tpb native", "-ndt 3", "--port 8000",
                "turboderp-Qwen3.8-27B-EXL3-SC4.0bpw-H5",   # the artifact since 2026-09-04
                "Mia-AiLab-Qwen3.8-27B-EXL3-3.5bpw"):        # named as the previous one
        assert tok in body, tok
    # the three positional knobs, in this order, with the measured defaults
    assert re.search(r'set CS=%~1\s+if "%CS%"=="" set CS=163840', body)
    assert re.search(r'set GS=%~2\s+if "%GS%"=="" set GS=9,15\.5', body)
    assert re.search(r'set HOST=%~3\s+if "%HOST%"=="" set HOST=127\.0\.0\.1', body)


def test_the_hub_offers_both_keys():
    body = read(HUB)
    for n in BATS:
        assert n in body, n
    m = re.search(r"choice /c (\S+) /n /m \"  Choose", body)
    keys = m.group(1)
    assert "F" in keys and "G" in keys and keys.endswith("Q")
    assert 'set "LOOP=serve-exl3.bat"' in body
    assert 'set "LOOP=serve-exl3-max.bat"' in body
    assert "197K measured" in body  # G says so on the menu (2026-09-04, results 10)
