"""Hidden oracle for openclink #144 — own the whole process tree (2026-09-24).

Black-box: drives BaseCLIAgent.run with a real fake CLI (this file, re-executed)
that spawns a grandchild writing a heartbeat file every 50 ms. Never shown to the
model. POSIX only (the sandbox is Linux); Windows Job Objects are out of scope.

What "owned" means here, from the issue's acceptance criteria:
- the CLI runs in its own process group / session (not the server's)
- every exit path (timeout, cancellation, success, failure) leaves no descendant
- run() returns / raises only after the tree is gone (confirmed, not requested)
- a grandchild that inherited the stdout pipe must not hang the timeout path
"""
from __future__ import annotations

import asyncio
import functools
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

GRANDCHILD = (
    "import sys,time\n"
    "p=sys.argv[1]\n"
    "while True:\n"
    "    open(p,'a').write(str(time.time())+'\\n'); time.sleep(0.05)\n"
)


def _fake_cli(mode: str, pidfile: str, beat: str, inherit: str) -> None:
    """Entry point when this file is executed as the fake CLI."""
    out = None if inherit == "inherit" else subprocess.DEVNULL
    child = subprocess.Popen([sys.executable, "-c", GRANDCHILD, beat],
                             stdout=out, stderr=out, stdin=subprocess.DEVNULL)
    Path(pidfile).write_text(f"{os.getpid()} {os.getpgrp()} {child.pid}")
    sys.stdin.read()
    if mode == "hang":
        time.sleep(3600)
    elif mode == "succeed":
        sys.stdout.write('{"response": "ok"}')
        sys.stdout.flush()
        sys.exit(0)
    elif mode == "fail":
        sys.stderr.write("boom")
        sys.exit(3)


if __name__ == "__main__" and len(sys.argv) == 5:
    _fake_cli(*sys.argv[1:])
    sys.exit(0)


from clink.agents.base import BaseCLIAgent, CLIAgentError  # noqa: E402
from clink.models import ResolvedCLIClient, ResolvedCLIRole  # noqa: E402

pytestmark = pytest.mark.skipif(os.name != "posix", reason="POSIX process groups")


def arun(fn):
    """The sandbox has no pytest-asyncio; run each async test on a fresh loop."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return asyncio.run(fn(*args, **kwargs))
    return wrapper


def _alive(pid: int) -> bool:
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except (FileNotFoundError, ProcessLookupError, IndexError):
        return False
    return state not in ("Z", "X")


def _group_alive(pgid: int) -> bool:
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
        except (FileNotFoundError, ProcessLookupError, IndexError):
            continue
        if int(fields[2]) == pgid and fields[0] not in ("Z", "X"):
            return True
    return False


class Tree:
    def __init__(self, tmp: Path, mode: str, inherit: str, timeout: int):
        self.pidfile = tmp / "pids"
        self.beat = tmp / "beat"
        prompt = tmp / "role.txt"
        prompt.write_text("role")
        role = ResolvedCLIRole(name="default", prompt_path=prompt, role_args=[])
        client = ResolvedCLIClient(
            name="fake", executable=[sys.executable],
            internal_args=[str(Path(__file__).resolve()), mode, str(self.pidfile), str(self.beat), inherit],
            config_args=[], env={}, timeout_seconds=timeout, parser="gemini_json",
            roles={"default": role}, output_to_file=None, working_dir=None)
        self.agent = BaseCLIAgent(client)
        self.role = role

    def run(self):
        return self.agent.run(role=self.role, prompt="go", files=[], images=[])

    def pids(self):
        cli, pgid, grand = map(int, self.pidfile.read_text().split())
        return cli, pgid, grand

    def growing(self) -> bool:
        before = self.beat.stat().st_size if self.beat.exists() else 0
        time.sleep(0.4)
        after = self.beat.stat().st_size if self.beat.exists() else 0
        return after > before

    def reap(self):  # never leak into the next test, whatever the result
        if self.pidfile.exists():
            cli, _pgid, grand = self.pids()
            # Never signal the pgid: on unfixed code it is the test's own group.
            for pid in (grand, cli):
                try:
                    os.kill(pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass


async def bounded(tree: Tree, awaitable, limit: float = 20):
    """Fail, never hang: on the unfixed code run() can block forever, and a live
    child would then also block the loop's shutdown. Kill the fake tree first."""
    try:
        return await asyncio.wait_for(awaitable, limit)
    except asyncio.TimeoutError:
        tree.reap()
        raise AssertionError(f"run() did not return within {limit} s")


async def _wait_for_pidfile(tree: Tree):
    for _ in range(200):
        if tree.pidfile.exists() and tree.pidfile.read_text().count(" ") == 2:
            return
        await asyncio.sleep(0.05)
    raise AssertionError("fake CLI never started")


@pytest.fixture
def make_tree(tmp_path):
    made = []

    def make(mode, inherit="devnull", timeout=2):
        where = tmp_path / f"t{len(made)}"
        where.mkdir()
        tree = Tree(where, mode, inherit, timeout)
        made.append(tree)
        return tree

    yield make
    for tree in made:
        tree.reap()


def _assert_tree_gone(tree: Tree):
    cli, pgid, grand = tree.pids()
    assert not _alive(grand), "grandchild survived"
    assert not _alive(cli), "CLI survived"
    assert not tree.growing(), "something is still writing"


@arun
async def test_cli_runs_in_its_own_process_group(make_tree):
    tree = make_tree("succeed")
    await bounded(tree, tree.run())
    _, pgid, _ = tree.pids()
    assert pgid != os.getpgrp(), "CLI shares the server's process group"


@arun
async def test_timeout_kills_the_grandchild(make_tree):
    tree = make_tree("hang")
    with pytest.raises(CLIAgentError):
        await bounded(tree, tree.run())
    _assert_tree_gone(tree)


@arun
async def test_timeout_does_not_hang_when_a_grandchild_holds_the_pipe(make_tree):
    tree = make_tree("hang", inherit="inherit")
    started = time.monotonic()
    with pytest.raises(CLIAgentError):
        await bounded(tree, tree.run())
    assert time.monotonic() - started < 15
    _assert_tree_gone(tree)


@arun
async def test_timeout_returns_only_after_the_group_is_gone(make_tree):
    tree = make_tree("hang")
    with pytest.raises(CLIAgentError):
        await bounded(tree, tree.run())
    _, pgid, _ = tree.pids()
    assert not _group_alive(pgid), "run() raised while the group was still alive"


@arun
async def test_cancellation_kills_the_tree_before_it_is_acknowledged(make_tree):
    tree = make_tree("hang", timeout=120)
    task = asyncio.ensure_future(tree.run())
    await _wait_for_pidfile(tree)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await bounded(tree, task)
    _, pgid, _ = tree.pids()
    assert not _group_alive(pgid), "cancellation acknowledged while the group lived"
    _assert_tree_gone(tree)


@arun
async def test_success_leaves_no_descendant(make_tree):
    tree = make_tree("succeed")
    await bounded(tree, tree.run())
    _assert_tree_gone(tree)


@arun
async def test_failure_leaves_no_descendant(make_tree):
    tree = make_tree("fail")
    with pytest.raises(CLIAgentError):
        await bounded(tree, tree.run())
    _assert_tree_gone(tree)
