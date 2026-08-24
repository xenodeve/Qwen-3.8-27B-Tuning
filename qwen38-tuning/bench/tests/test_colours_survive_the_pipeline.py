r"""llama.cpp's own colours, kept alive through a pipeline that would kill them.

WHY THEY WERE MISSING (issue #49). `serve.ps1` reads the profile's output line by
line so it can check residency and print the status inline. That pipeline is not
a terminal, and llama.cpp's `--log-colors` defaults to **auto**, which means
"colour when stdout is a TTY". Piped, it saw no TTY and turned colour off.

Nothing was broken and nothing said so -- the output was simply plainer than the
same server run by hand, which is the kind of difference that gets blamed on the
wrong thing.

THE FIX IS A FLAG, AND IT LIVES IN THE PROFILE.

`--log-colors on` makes llama.cpp emit ANSI regardless of what it is writing to.
It is a serving flag, so it belongs in `worker-q2kxl-mtp.ps1` with every other
one; `serve.ps1` asks for it by parameter, the same way it asks for verbosity and
the bind address. A copy in the launcher would be the second source of truth this
whole arrangement exists to avoid.

THE DEFAULT DOES NOT MOVE. The profile defaults to `auto`, which is exactly what
llama.cpp does when the flag is absent -- so a profile run by hand behaves as it
always did, and no measured row changes meaning.

WHAT THIS FILE CANNOT CHECK. That the colours render. That needs a console and
an eye; it is verified by capturing the stream and finding ESC[ sequences in it,
which is done by running the thing, not from pytest.
"""
import os
import re

BENCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(BENCH))
SERVE = os.path.join(ROOT, "serve.ps1")
PROFILE = os.path.join(ROOT, "qwen38-tuning", "scripts", "worker-q2kxl-mtp.ps1")


def read(path):
    with open(path, encoding="utf-8", errors="replace") as fh:
        return fh.read()


def test_the_profile_owns_the_flag():
    p = read(PROFILE)
    assert "--log-colors $LogColors" in p, (
        "the profile does not pass the colour flag from its parameter")


def test_the_profile_default_matches_llama_cpp_s_own():
    """`auto` is what llama.cpp does with the flag absent. Anything else would
    change how the profile behaves when run by hand, and every measured row was
    taken that way."""
    p = read(PROFILE)
    assert re.search(r"\$LogColors\s*=\s*['\"]auto['\"]", p), (
        "the profile's colour default is not auto")


def test_the_launcher_asks_for_colour_rather_than_declaring_it():
    """Same rule as -Verbosity and -BindAddress: the flag lives in one file."""
    s = read(SERVE)
    # Code lines only. The first version matched the whole file and went red on
    # a COMMENT that named the flag while explaining why the launcher does not
    # declare it -- the third time in this suite a test has failed on prose
    # rather than on behaviour.
    code = [ln for ln in s.splitlines()
            if not ln.strip().startswith('#') and '--log-colors' in ln]
    assert not code, (
        "serve.ps1 declares the flag itself; it must ask the profile: %r" % code)
    assert "LogColors" in s, "serve.ps1 never requests colour"


def test_the_launcher_asks_for_it_on():
    s = read(SERVE)
    assert re.search(r"LogColors'?\]?\s*=\s*'on'", s), (
        "serve.ps1 does not turn colour on, so the pipeline still suppresses it")


def test_the_launcher_does_not_rely_on_powershell_s_default_rendering():
    """MEASURED: with the flag on, llama.cpp emits 1,180 escape bytes in four
    codes -- blue timestamps, green INFO, magenta warnings, reset. They vanished
    from a capture anyway, because PowerShell 7 strips ANSI at render time
    whenever output is not a console: $PSStyle.OutputRendering was PlainText.

    So the colours were being removed by the thing forwarding them, one layer
    past where anyone would look. Setting OutputRendering to Ansi passes them
    through verbatim regardless of what the output is attached to, which is what
    llama.cpp itself does when told --log-colors on."""
    s = read(SERVE)
    assert re.search(r"OutputRendering\s*=\s*'Ansi'", s), (
        "serve.ps1 leaves ANSI rendering to a default that strips it whenever "
        "the output is redirected")


def test_the_pipeline_does_not_strip_what_it_forwards():
    """Write-Host passes ANSI through; formatting or -join would not. The line
    must reach the console as it arrived."""
    s = read(SERVE)
    assert re.search(r"Write-Host\s+\$line\b", s), (
        "the forwarded line is reformatted on its way to the console")
