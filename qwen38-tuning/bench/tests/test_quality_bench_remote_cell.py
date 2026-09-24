"""A remote cell runs against a gateway and must never touch a local server.

INCIDENT THIS GUARDS. `ensure_server` dispatched on `if backend == "exl3": ...
else: <boot llama-server>`. The `else` was not "the other remote thing" -- it
was llama.cpp, and it begins by calling `stop_exl3(); stop_llama()` and then
`start_hidden(serve-dual-nvfp4-deep.bat)`. So adding ANY new backend to CELLS
without touching that branch means the new cell **kills the running servers and
boots the GPU**, which on 2026-09-07 is exactly what the developer had asked not
to happen ("ยังไม่อยากเปิด server").

The failure would not have announced itself: the gateway cell would still have
answered, the run would still have produced a page, and the only symptom would
have been a GPU that was busy for ten minutes for no reason.

So the dispatch is now explicit per backend and an unknown backend raises. Two
of the tests below are about the branch that does NOT run.
"""
import importlib.util
import json
import os
import sys

import pytest

BENCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TUNING = os.path.dirname(BENCH)

_spec = importlib.util.spec_from_file_location(
    "quality_bench", os.path.join(TUNING, "tools", "quality-bench.py"))
qb = importlib.util.module_from_spec(_spec)
sys.modules["quality_bench"] = qb
_spec.loader.exec_module(qb)


@pytest.fixture
def no_server_may_be_touched(monkeypatch):
    """Any call that would stop or start a local server is a test failure."""
    def boom(*a, **k):
        raise AssertionError("a local server was touched by a remote cell")
    monkeypatch.setattr(qb, "stop_exl3", boom)
    monkeypatch.setattr(qb, "stop_llama", boom)
    monkeypatch.setattr(qb, "start_hidden", boom)
    monkeypatch.setattr(qb, "get", lambda *a, **k: None)
    return boom


def test_the_gateway_cell_exists_and_names_the_model_the_gateway_reports():
    """Measured 2026-09-07: the assistant message came back as this string.

    Pinned as a literal because the profile asks for `qwen3.8-27b-fp8` and the
    client logs `unrecognized_model` -- what the gateway actually serves is the
    only name worth recording.
    """
    d = qb.CELLS["D"]
    assert d["backend"] == "remote"
    assert d["model"] == "vllm/Qwen/Qwen3.8-27B-FP8"
    assert d["base"] == "https://gateway.9arm.co"


def test_a_remote_cell_starts_and_stops_nothing(no_server_may_be_touched):
    qb.ensure_server("D")          # must return without touching a server


def test_an_unknown_backend_raises_instead_of_booting_the_gpu(no_server_may_be_touched, monkeypatch):
    """The bug this file exists for: `else` used to mean llama-server."""
    monkeypatch.setitem(qb.CELLS, "ZZ", dict(qb.CELLS["D"], backend="something-new"))
    with pytest.raises(RuntimeError, match="backend"):
        qb.ensure_server("ZZ")


def test_the_gateway_token_is_read_from_the_profile_not_hardcoded():
    """The bench sends `sk-local` to local servers; a gateway needs the real one,
    and it must come from the profile file rather than the source."""
    src = open(os.path.join(TUNING, "tools", "quality-bench.py"), encoding="utf-8").read()
    assert "sk-ant" not in src and "gateway.9arm.co/" not in src.replace('"https://gateway.9arm.co"', "")
    prof = qb.CELLS["D"]["profile"]
    assert os.path.basename(prof) == ".claude-9arm.json"
    if os.path.exists(prof):
        want = json.load(open(prof, encoding="utf-8"))["env"]["ANTHROPIC_AUTH_TOKEN"]
        assert qb.auth_token(qb.CELLS["D"]) == want
    assert qb.auth_token(qb.CELLS["A"]) == "sk-local"


def test_the_bench_does_not_borrow_the_profiles_hooks(monkeypatch, tmp_path):
    """The 9arm profile carries a SessionStart hook that injects the whole
    `using-qwen38` router. Passing `--settings <profile>` would put that router
    into EVERY arm including `noskill`, which is arm contamination, not a bench.
    The cell therefore supplies base+token as env and lets CLAUDE_CONFIG_DIR
    alone decide what skills and hooks the session sees."""
    src = open(os.path.join(TUNING, "tools", "quality-bench.py"), encoding="utf-8").read()
    assert "--settings" not in src


def test_a_remote_timeout_does_not_restart_a_local_server():
    """The timeout path mirrors ensure_server's dispatch and had the same `else`."""
    src = open(os.path.join(TUNING, "tools", "quality-bench.py"), encoding="utf-8").read()
    i = src.index("orphaned generation")
    window = src[i:i + 400]
    assert 'backend"] == "remote"' in window or "is_remote" in window, (
        "the timeout handler must skip the server restart for a remote cell")


def test_remote_server_evidence_is_absent_rather_than_invented():
    ev = qb.server_evidence(qb.CELLS["D"], "2026-09-07T00:00:00+00:00",
                            "2026-09-07T00:01:00+00:00", str(BENCH))
    assert ev.get("server_evidence") in (None, "n/a: remote gateway, no local server log")
    assert "requests" not in ev


# ---------------------------------------------------------------------------
# Round 1, 2026-09-07. The first D run returned rc=1 in 0.0 min with 0 output
# tokens -- the gateway answered 403 -- and the runner printed
# `gate=1/5  hidden 1 passed / 7 failed`. It scored a run that never reached the
# model. That is the instrument fault this repo exists to refuse, so the guard
# below came before anything else was measured.
# ---------------------------------------------------------------------------

def test_the_cell_sends_the_name_the_gateway_accepts_not_the_name_it_reports():
    """403: `This team can only access models=['qwen3.8-27b-fp8']. Tried to access
    vllm/Qwen/Qwen3.8-27B-FP8`. The request name and the served name are two
    different strings and both are worth keeping -- CORRECTIONS #34's shape."""
    d = qb.CELLS["D"]
    assert d["request_model"] == "qwen3.8-27b-fp8"
    assert d["model"] == "vllm/Qwen/Qwen3.8-27B-FP8"
    src = open(os.path.join(TUNING, "tools", "quality-bench.py"), encoding="utf-8").read()
    assert 'c.get("request_model", c["model"])' in src, "the argv must send the request name"


def test_a_run_that_errored_is_void_rather_than_scored():
    assert qb.void_reason({"is_error": True, "final_text": "403 team not allowed"},
                          rc=1, timed_out=False)


def test_a_run_with_no_output_tokens_is_void_rather_than_scored():
    assert qb.void_reason({"is_error": False, "output_tokens": 0}, rc=1, timed_out=False)


def test_a_timeout_is_not_voided_by_the_token_rule():
    """A timed-out run really did work; it is a different outcome from never starting."""
    assert qb.void_reason({"is_error": False, "output_tokens": 0}, rc=-1, timed_out=True) is None


def test_a_real_run_is_not_void():
    assert qb.void_reason({"is_error": False, "output_tokens": 4321}, rc=0, timed_out=False) is None


def test_the_run_records_the_model_the_server_actually_served():
    """Otherwise the `model` column is an assertion about provenance that nothing checks."""
    src = open(os.path.join(TUNING, "tools", "quality-bench.py"), encoding="utf-8").read()
    assert "served_model" in src


# ---------------------------------------------------------------------------
# Round 2, 2026-09-07. Both D cells reported `test_runs_seen: 3` and
# `red_then_green: False`. The stream says the model ran pytest ONCE; the other
# two "test runs" were Read results of source files. The selector matched a bare
# "Error" anywhere in the text, and `ledger/parse.py` raises ValueError -- so
# check 3 ("a test command ran") passed for free on every run of this fixture,
# without a test having run at all.
#
# Second bug in the same function: selection read the FULL text, classification
# read only text[-300:], so those entries took an index and voted neither red nor
# green.
# ---------------------------------------------------------------------------

def _stream(tmp_path, events):
    p = tmp_path / "stream.jsonl"
    p.write_text("\n".join(json.dumps(e) for e in events), encoding="utf-8")
    return str(p)


def _use(tid, name):
    return {"type": "assistant", "message": {"content": [{"type": "tool_use", "id": tid, "name": name, "input": {}}]}}


def _result(tid, text):
    return {"type": "user", "message": {"content": [{"type": "tool_result", "tool_use_id": tid, "content": text}]}}


READ_OF_A_FILE_THAT_RAISES = (
    "    28\tdef parse_line(raw):\n    29\t        raise ValueError(f'bad line: {raw}')\n"
    + "x" * 400 + "\n    35\t    return entries\n")


def test_reading_a_file_that_mentions_ValueError_is_not_a_test_run(tmp_path):
    s = _stream(tmp_path, [_use("t1", "Read"), _result("t1", READ_OF_A_FILE_THAT_RAISES)])
    assert qb.test_command_outputs(s) == []


def test_a_bash_pytest_summary_is_a_test_run(tmp_path):
    s = _stream(tmp_path, [_use("t1", "Bash"), _result("t1", "..........   [100%]\n10 passed in 0.06s")])
    assert len(qb.test_command_outputs(s)) == 1


def test_the_read_noise_no_longer_hides_a_real_red_then_green(tmp_path):
    """The shape the D cells would have produced had the model done TDD."""
    s = _stream(tmp_path, [
        _use("t0", "Read"),   _result("t0", READ_OF_A_FILE_THAT_RAISES),
        _use("t1", "Bash"),   _result("t1", "1 failed, 9 passed in 0.04s"),
        _use("t2", "Bash"),   _result("t2", "10 passed in 0.05s"),
    ])
    outs = qb.test_command_outputs(s)
    assert len(outs) == 2
    assert qb.red_before_green(outs) is True


def test_green_only_is_not_red_then_green(tmp_path):
    s = _stream(tmp_path, [_use("t1", "Bash"), _result("t1", "10 passed in 0.05s")])
    assert qb.red_before_green(qb.test_command_outputs(s)) is False


def test_green_then_red_is_not_red_then_green(tmp_path):
    s = _stream(tmp_path, [
        _use("t1", "Bash"), _result("t1", "10 passed in 0.05s"),
        _use("t2", "Bash"), _result("t2", "2 failed, 8 passed in 0.05s"),
    ])
    assert qb.red_before_green(qb.test_command_outputs(s)) is False


def test_a_failure_far_from_the_end_of_a_long_output_is_still_a_failure(tmp_path):
    """The truncation bug: selection saw the whole text, classification saw the last 300."""
    long_red = "1 failed, 9 passed in 0.04s\n" + "trailing detail\n" * 60
    s = _stream(tmp_path, [
        _use("t1", "Bash"), _result("t1", long_red),
        _use("t2", "Bash"), _result("t2", "10 passed in 0.05s"),
    ])
    assert qb.red_before_green(qb.test_command_outputs(s)) is True


def test_regate_refuses_a_void_cell_too():
    """Fix the class, not the instance (traps.md #11). `void_reason` was added to run_cell
    and --regate scored the dead 403 cell 3/5 anyway, from the same summary.json that
    already said rc=1 and output_tokens=0. Both paths consult one function."""
    src = open(os.path.join(TUNING, "tools", "quality-bench.py"), encoding="utf-8").read()
    body = src[src.index("if a.regate:"):]
    assert "void_reason(" in body, "the regate loop must apply the same void test"


def test_void_reason_reads_a_stored_summary_without_extra_arguments():
    """regate has only summary.json, so the same call must work from stored fields."""
    stored = {"is_error": True, "final_text": "403", "rc": 1, "timed_out": False}
    assert qb.void_reason(stored, stored.get("rc"), stored.get("timed_out"))


# ---------------------------------------------------------------------------
# Round 3, 2026-09-07. `red_then_green` does not measure TDD.
#
# Across 20 paired code2 runs it was True 7/10 and 9/10 -- yet in EVERY one of
# those 20 runs the model wrote `parse.py`, `report.py`, `__init__.py` and
# `README.md` before it wrote a single line of test. What `red_then_green`
# actually detects is "a test failed at some point and passed later", which is
# what ordinary iteration looks like when the first implementation is buggy.
#
# The property the skill asks for is an ORDER: a test run before the source is
# touched. Measured that way the same 20 runs give codegate 5/10 and codegate2
# 1/10 -- the opposite ranking to the gate score.
# ---------------------------------------------------------------------------

def _tool(tid, name, path=None, turn_content=None):
    inp = {"file_path": path} if path else {}
    return {"type": "assistant", "message": {"content": [{"type": "tool_use", "id": tid, "name": name, "input": inp}]}}


def test_a_test_run_before_any_source_edit_is_tdd_order(tmp_path):
    s = _stream(tmp_path, [
        _tool("t1", "Write", "/w/tests/test_x.py"),
        _tool("t2", "Bash"), _result("t2", "1 failed, 2 passed in 0.01s"),
        _tool("t3", "Edit", "/w/pkg/parse.py"),
    ])
    r = qb.tdd_order(s)
    assert r["tested_before_source_edit"] is True
    assert r["first_source_edit_turn"] == 3
    assert r["first_test_run_turn"] == 2


def test_editing_the_source_first_is_not_tdd_order(tmp_path):
    """The shape all 20 runs actually had."""
    s = _stream(tmp_path, [
        _tool("t1", "Edit", "/w/ledger/parse.py"),
        _tool("t2", "Write", "/w/README.md"),
        _tool("t3", "Write", "/w/tests/test_ledger.py"),
        _tool("t4", "Bash"), _result("t4", "10 passed in 0.05s"),
    ])
    r = qb.tdd_order(s)
    assert r["tested_before_source_edit"] is False
    assert r["first_source_edit_turn"] == 1


def test_a_readme_is_not_a_source_file(tmp_path):
    """Docs are neither test nor implementation; counting README as source would
    fail a run that documented first and still did TDD."""
    s = _stream(tmp_path, [
        _tool("t1", "Write", "/w/README.md"),
        _tool("t2", "Write", "/w/tests/test_x.py"),
        _tool("t3", "Bash"), _result("t3", "1 failed in 0.01s"),
        _tool("t4", "Edit", "/w/pkg/parse.py"),
    ])
    r = qb.tdd_order(s)
    assert r["tested_before_source_edit"] is True


def test_a_test_file_is_not_a_source_file(tmp_path):
    s = _stream(tmp_path, [_tool("t1", "Write", "/w/tests/test_ledger.py")])
    assert qb.tdd_order(s)["first_source_edit_turn"] is None


def test_no_test_run_at_all_is_not_tdd_order(tmp_path):
    s = _stream(tmp_path, [_tool("t1", "Edit", "/w/pkg/parse.py")])
    assert qb.tdd_order(s)["tested_before_source_edit"] is False


def test_the_code_gate_reports_the_order_alongside_red_then_green():
    """Both are kept: red_then_green stays comparable with the 2026-09-05 baseline,
    and the order is the one that measures what the skill asks for."""
    src = open(os.path.join(TUNING, "tools", "quality-bench.py"), encoding="utf-8").read()
    assert "tested_before_source_edit" in src and "tdd_order(" in src


# ---------------------------------------------------------------------------
# Round 4, 2026-09-07. The D cells measured nothing about skills: the `family`
# arm called the Skill tool ZERO times on the gateway, while the same arm on
# EXL3 called it 4x in every rep. The skills were registered either way -- both
# streams' init events list 48 slash commands including all four qwen ones, and
# `Skill` is in `tools`. Asked in words ("ใช้ Skill tool โหลด qwen38-code-gate"),
# the gateway model loads the file and quotes its first heading correctly.
#
# So the slash token in a -p prompt is what does not carry. A cell can ask for
# the tool by name instead; A keeps the token so the 2026-09-05 baseline stays
# comparable, and the difference is recorded on the cell.
# ---------------------------------------------------------------------------

def test_a_normal_cell_keeps_the_arms_own_suffix():
    spec = {"skills": ["qwen38-code-gate"], "suffix": " | จบด้วย /qwen38-code-gate"}
    assert qb.brief_suffix(qb.CELLS["A"], spec) == spec["suffix"]


def test_an_explicit_cell_names_the_skill_tool_and_every_skill():
    spec = {"skills": ["using-qwen38", "qwen38-think"], "suffix": " | /using-qwen38"}
    out = qb.brief_suffix(qb.CELLS["D"], spec)
    assert "Skill" in out
    assert "/using-qwen38" in out and "/qwen38-think" in out


def test_an_explicit_cell_with_no_skills_is_left_alone():
    """`noskill` must stay a true control -- nothing to load, nothing added."""
    spec = {"skills": [], "suffix": ""}
    assert qb.brief_suffix(qb.CELLS["D"], spec) == ""


def test_the_gateway_cell_is_marked_explicit_and_exl3_is_not():
    assert qb.CELLS["D"].get("skills_explicit") is True
    assert not qb.CELLS["A"].get("skills_explicit")


# ---------------------------------------------------------------------------
# 2026-09-08, second half. `brief_suffix` was wired into the CODE path only.
# Page arms build their brief from ARMS[arm]["brief"], which embeds slash tokens
# ("/using-design  /ask-xeno | จบด้วย /design-ship-gate") -- exactly the form that
# produced zero Skill calls on the gateway. A frontend run on D would have
# repeated the fault the code path had already been fixed for.
# ---------------------------------------------------------------------------

def test_a_page_arm_on_an_explicit_cell_asks_for_the_skill_tool():
    spec = {"skills": ["using-design", "design-ship-gate"], "brief": "B | /using-design"}
    out = qb.page_brief(qb.CELLS["D"], spec)
    assert "Skill" in out
    assert "/using-design" in out and "/design-ship-gate" in out


def test_a_page_arm_on_a_normal_cell_is_unchanged():
    spec = {"skills": ["using-design"], "brief": "B | /using-design"}
    assert qb.page_brief(qb.CELLS["A"], spec) == "B | /using-design"


def test_a_page_noskill_arm_is_never_rewritten():
    spec = {"skills": [], "brief": "B"}
    assert qb.page_brief(qb.CELLS["D"], spec) == "B"


def test_the_whole_user_skills_arm_is_not_rewritten():
    """`skill` uses the developer's entire ~/.claude (skills=None); there is no list to name."""
    spec = {"skills": None, "brief": "B | /using-design"}
    assert qb.page_brief(qb.CELLS["D"], spec) == "B | /using-design"


# ---------------------------------------------------------------------------
# 2026-09-09. `D-designonly-r4` scored 6/8 with 0 Thai and no dark mode, and its
# transcript shows `Skill` called ZERO times -- two turns, one Write, 2.7 min. It
# never read the skill it was supposed to be testing. Written up first as "the
# model ignored the rules", which is a different fault with a different fix.
#
# Every one of the other twelve cells loaded the skill, and every one of those
# followed the rules that existed when it ran (language 7/7, dark 3/3). So an arm
# cell that never loaded its skills is not a measurement of that arm, and must
# say so rather than contribute a number.
# ---------------------------------------------------------------------------

def test_a_skill_arm_that_never_called_skill_is_flagged():
    summ = {"tool_calls": {"Write": 1}, "output_tokens": 16172}
    assert qb.skills_not_loaded(summ, ["using-design", "design-rules"])


def test_a_skill_arm_that_called_skill_is_not_flagged():
    summ = {"tool_calls": {"Skill": 7, "Write": 1}, "output_tokens": 28725}
    assert qb.skills_not_loaded(summ, ["using-design"]) is None


def test_a_noskill_arm_is_never_flagged():
    """`noskill` has nothing to load; absence of Skill is correct there."""
    assert qb.skills_not_loaded({"tool_calls": {"Write": 1}}, []) is None
    assert qb.skills_not_loaded({"tool_calls": {"Write": 1}}, None) is None


def test_the_flag_is_recorded_on_the_summary_not_thrown_away():
    src = open(os.path.join(TUNING, "tools", "quality-bench.py"), encoding="utf-8").read()
    assert "skills_not_loaded" in src
    body = src[src.index("def run_cell"):]
    assert 'summary["skills_not_loaded"]' in body
