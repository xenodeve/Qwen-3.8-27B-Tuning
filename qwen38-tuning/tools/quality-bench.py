"""Three-artifact quality bench through Claude Code (2026-09-05, AFK batch).

One cell = one artifact x one arm x one repeat. The same Thai brief the developer
used on 2026-09-03/04 is given to `claude -p`; the page it writes is kept with the
whole stream-json transcript, the server evidence for the window, screenshots,
and the eight design-ship-gate checks run by this script (not by the model).

  python qwen38-tuning\\tools\\quality-bench.py --queue A:noskill B:noskill C:noskill A:skill --deadline 11:35

Artifacts (each at its profile's largest window):
  A  EXL3 turboderp SC 4.0bpw H5      262,144   :8000   launchers\\serve-exl3-max.bat
  B  EXL3 Mia-AiLab 3.5bpw            262,144   :8000   same, EXL3_MODEL_DIR
  C  llama.cpp NVFP4-LOW (esatapedico) 200,704  :8080   launchers\\serve-dual-nvfp4-deep.bat

Arms:
  noskill  CLAUDE_CONFIG_DIR = a fresh dir with settings only -> no user skills at all;
           brief without the slash tokens
  skill    the user's own ~/.claude (324 skills incl. the design suite and
           design-ship-gate); brief exactly as the developer typed it, plus
           "จบด้วย /design-ship-gate"

Known confound, stated: llama-server's Anthropic route ignores effort, so C runs the
template default xhigh while A/B run --effort medium.
"""
import argparse, datetime, glob, json, os, re, shutil, subprocess, sys, time, urllib.request

ROOT = r"C:\AI"
RESULTS = os.path.join(ROOT, "qwen38-tuning", "results", "quality-2026-09-05")
WORK_ROOT = os.environ.get("QBENCH_WORK_ROOT", r"D:\qbench-work")   # no CLAUDE.md in any ancestor
GATE_DIR = os.path.join(os.path.expanduser("~"), ".claude", "skills", "design-ship-gate")
NODE_PATH = r"C:\Users\xenod\AppData\Local\npm-cache\_npx\361ceb562f3b3235\node_modules"
CLAUDE = shutil.which("claude.cmd") or shutil.which("claude") or "claude"
RUN_CAP_S = 75 * 60

BRIEF = "สร้าง Landing Page แนะนำ Google โดยใช้ Editorial Minimalism × Modern Swiss × Liquid Glass × Visible Grid | ทำเป็น HTML ไฟล์เดียว"
BRIEF_SKILL = BRIEF + " | /using-design  /ask-xeno | จบด้วย /design-ship-gate"

# Isolated skill sets (2026-09-05 afternoon, the developer's question: which KIND of skill
# moves this model). Each arm gets its own CLAUDE_CONFIG_DIR holding only the listed skills,
# so nothing else (the other 300 skills, the global CLAUDE.md) can reach the session.
DESIGN_SUITE = ["using-design", "design-setup", "design-rules", "design-audit", "design-psychology", "ask-xeno"]
ARMS = {
    "noskill":    dict(skills=[], brief=BRIEF),
    "gateonly":   dict(skills=["design-ship-gate"], brief=BRIEF + " | จบด้วย /design-ship-gate"),
    "designonly": dict(skills=DESIGN_SUITE, brief=BRIEF + " | /using-design  /ask-xeno"),
    "both":       dict(skills=DESIGN_SUITE + ["design-ship-gate"], brief=BRIEF_SKILL),
    "skill":      dict(skills=None, brief=BRIEF_SKILL),   # the developer's whole ~/.claude
}
USER_SKILLS = os.path.join(os.path.expanduser("~"), ".claude", "skills")

# Task types. "page" is the Google landing brief above; "code1" is fixtures/code-task-1
# (xeno-skills #356, slice 1 of #355): a planted bug + a small feature, hidden tests copied
# in AFTER the run, the gate computed here. Code arms carry their own brief suffixes.
FIXTURES = os.path.join(ROOT, "qwen38-tuning", "fixtures")
TASK_FIXTURE = {"code1": "code-task-1", "code2": "code-task-2", "think1": "think-task-1", "think2": "think-task-2", "think3": "think-task-3", "think4": "think-task-4"}
CODE_ARMS = {
    "noskill":  dict(skills=[], suffix=""),
    "codegate": dict(skills=["qwen38-code-gate"], suffix=" | จบด้วย /qwen38-code-gate"),
    "think":    dict(skills=["qwen38-think"], suffix=" | เริ่มด้วย /qwen38-think"),
    "family":   dict(skills=["using-qwen38", "qwen38-think", "qwen38-code-gate", "karpathy-guidelines"], suffix=" | /using-qwen38"),
}

CELLS = {
    "A": dict(backend="exl3", model_dir=r"C:\AI\models\turboderp-Qwen3.8-27B-EXL3-SC4.0bpw-H5",
              model="turboderp-Qwen3.8-27B-EXL3-SC4.0bpw-H5", base="http://127.0.0.1:8000", ctx=262144, expect_min=28),
    "B": dict(backend="exl3", model_dir=r"C:\AI\models\Mia-AiLab-Qwen3.8-27B-EXL3-3.5bpw",
              model="Mia-AiLab-Qwen3.8-27B-EXL3-3.5bpw", base="http://127.0.0.1:8000", ctx=262144, expect_min=28),
    "C": dict(backend="llama", model="Qwen3.8-27B-NVFP4-MTP", base="http://127.0.0.1:8080", ctx=200704, expect_min=40),
}


def _rmtree(path):
    """rmtree that survives git's read-only object files on Windows (WinError 5 at 16:40)."""
    import stat
    def onerror(func, p, exc):
        try:
            os.chmod(p, stat.S_IWRITE); func(p)
        except Exception:
            pass
    shutil.rmtree(path, onerror=onerror)


def log(msg):
    line = f"{datetime.datetime.now():%H:%M:%S} {msg}"
    print(line, flush=True)
    with open(os.path.join(RESULTS, "RUN.log"), "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def get(url, timeout=3):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception:
        return None


def ps(cmd):
    return subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True)


def stop_exl3():
    # stop-exl3.cmd writes exl3-stop.flag FIRST and the relaunch loop consumes it when
    # its Tee pipeline ends -- which can be more than 3 s after the kill. Deleting the
    # flag here raced that read on 2026-09-05 10:49 and the loop relaunched the default
    # model on top of the llama-server boot. Never delete the flag; serve-exl3.cmd
    # clears a stale one at its own start.
    subprocess.run(["cmd", "/c", os.path.join(ROOT, "qwen38-tuning", "scripts", "stop-exl3.cmd")], capture_output=True)
    deadline = time.time() + 60
    while time.time() < deadline and get("http://127.0.0.1:8000/health"):
        time.sleep(2)
    time.sleep(5)


def stop_llama():
    ps("Get-Process llama-server -ErrorAction SilentlyContinue | Stop-Process -Force")
    time.sleep(3)


def start_hidden(bat, env_extra=None):
    env = " ".join(f"$env:{k}='{v}';" for k, v in (env_extra or {}).items())
    ps(f"{env} Start-Process -FilePath $env:ComSpec -ArgumentList '/c','{bat}' -WorkingDirectory 'C:\\AI' -WindowStyle Hidden")


def ensure_server(cell):
    c = CELLS[cell]
    if c["backend"] == "exl3":
        h = get(c["base"] + "/health")
        if h and h.get("ok") and h.get("model") == c["model"] and h.get("context_length") == c["ctx"]:
            log(f"server ready already: {h['model']} @ {h['context_length']}")
            return
        stop_llama(); stop_exl3()
        log(f"starting EXL3 {c['model']} @ 262144")
        start_hidden(r"C:\AI\launchers\serve-exl3-max.bat", {"EXL3_MODEL_DIR": c["model_dir"]})
        deadline = time.time() + 8 * 60
        while time.time() < deadline:
            h = get(c["base"] + "/health")
            if h and h.get("ok") and h.get("model") == c["model"]:
                log(f"server ready: {h['model']} @ {h['context_length']}")
                return
            time.sleep(5)
        raise RuntimeError("EXL3 server did not come up")
    else:
        h = get(c["base"] + "/health")
        if h and h.get("status") == "ok":
            log("llama-server ready already")
            return
        stop_exl3(); stop_llama()
        log("starting llama-server NVFP4 deep (200,704)")
        start_hidden(r"C:\AI\launchers\serve-dual-nvfp4-deep.bat")
        deadline = time.time() + 10 * 60
        while time.time() < deadline:
            h = get(c["base"] + "/health")
            if h and h.get("status") == "ok":
                log("llama-server ready")
                return
            time.sleep(5)
        raise RuntimeError("llama-server did not come up")


def nvsmi():
    r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,memory.total", "--format=csv,noheader"], capture_output=True, text=True)
    return r.stdout.strip()


def newest(pattern):
    files = sorted(glob.glob(pattern), key=os.path.getmtime)
    return files[-1] if files else None


def run_cell(cell, arm, rep, task="page"):
    c = CELLS[cell]
    name = f"{cell}-{arm}-r{rep}" if task == "page" else f"{cell}-{task}-{arm}-r{rep}"
    d = os.path.join(RESULTS, name)
    work = os.path.join(WORK_ROOT, name)
    if os.path.exists(work):
        _rmtree(work)
    if os.path.exists(d):
        if os.path.exists(os.path.join(d, "summary.json")):
            # a finished cell is evidence; the 12:30 rep-1 queue deleted the morning's
            # A-noskill-r1 this way. Move it aside instead.
            shutil.move(d, d + "-superseded-" + datetime.datetime.now().strftime("%H%M%S"))
        else:
            _rmtree(d)
    os.makedirs(d, exist_ok=True)        # the results dir no longer contains work/, so make it itself
    os.makedirs(work)
    ensure_server(cell)

    env = os.environ.copy()
    env.update(ANTHROPIC_BASE_URL=c["base"], ANTHROPIC_AUTH_TOKEN="sk-local", NODE_PATH=NODE_PATH,
               CLAUDE_CODE_MAX_CONTEXT_TOKENS=str(c["ctx"]), CLAUDE_CODE_AUTO_COMPACT_WINDOW=str(c["ctx"]),
               CLAUDE_AUTOCOMPACT_PCT_OVERRIDE="95", API_TIMEOUT_MS="3600000",
               CLAUDE_CODE_MAX_OUTPUT_TOKENS=str(c["ctx"] if c["backend"] == "exl3" else 32768))
    if task == "page":
        spec = ARMS[arm]
        brief = spec["brief"]
    else:
        fixture = os.path.join(FIXTURES, TASK_FIXTURE[task])
        spec = CODE_ARMS[arm]
        # ONE line: claude.cmd on Windows cuts an argument at a newline, and everything after it
        # (the rest of the brief AND the flags) is lost -- the 15:53 noskill cell saw only the
        # brief's first line and answered "which two tasks?" as plain text.
        raw = open(os.path.join(fixture, "BRIEF.md"), encoding="utf-8").read()
        brief = " ".join(l.strip() for l in raw.splitlines() if l.strip()) + spec["suffix"]
        # seed the fixture into work/ as a git checkout (the gate's scope check diffs against
        # HEAD); hidden/ stays out until the run is over
        for item in os.listdir(fixture):
            if item in ("hidden", "BRIEF.md", "EXPECT.json", "__pycache__", ".pytest_cache"):
                continue
            src = os.path.join(fixture, item)
            (shutil.copytree if os.path.isdir(src) else shutil.copy)(src, os.path.join(work, item))
        git = ["git", "-c", "user.email=bench@local", "-c", "user.name=bench"]
        subprocess.run(git + ["init", "-q"], cwd=work, capture_output=True)
        subprocess.run(git + ["add", "."], cwd=work, capture_output=True)
        subprocess.run(git + ["commit", "-q", "-m", "fixture"], cwd=work, capture_output=True)
    if spec["skills"] is None:
        env.pop("CLAUDE_CONFIG_DIR", None)
    else:
        cfg = os.path.join(RESULTS, f"config-{arm}")
        os.makedirs(cfg, exist_ok=True)
        with open(os.path.join(cfg, "settings.json"), "w", encoding="utf-8") as fh:
            json.dump({"effortLevel": "medium", "showThinkingSummaries": True}, fh)
        sk = os.path.join(cfg, "skills")
        if os.path.isdir(sk):
            _rmtree(sk)
        os.makedirs(sk)
        for skill_name in spec["skills"]:          # not `name`: that is the cell's label (log lines said "design-ship-gate: start" on 2026-09-05)
            src = os.path.join(USER_SKILLS, skill_name)
            if not os.path.isdir(src):
                raise RuntimeError(f"skill {skill_name} not installed under {USER_SKILLS}")
            shutil.copytree(src, os.path.join(sk, skill_name))
        env["CLAUDE_CONFIG_DIR"] = cfg
    cmd = [CLAUDE, "-p", brief, "--model", c["model"], "--strict-mcp-config", "--effort", "medium",
           "--dangerously-skip-permissions", "--output-format", "stream-json", "--verbose", "--include-partial-messages"]

    log(f"== {name}: start ({c['model']} @ {c['ctx']}, arm {arm})")
    before = nvsmi()
    t0 = time.time(); t0_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
    timed_out = False
    with open(os.path.join(d, "stream.jsonl"), "w", encoding="utf-8") as out, open(os.path.join(d, "stderr.txt"), "w", encoding="utf-8") as err:
        try:
            p = subprocess.run(cmd, cwd=work, env=env, stdout=out, stderr=err, timeout=RUN_CAP_S)
            rc = p.returncode
        except subprocess.TimeoutExpired:
            timed_out, rc = True, -1
            # the client is dead but the server keeps generating for it (a 128K-token think on
            # 2026-09-05 13:19-14:05); the next cell would queue behind that orphan. Restart.
            log(f"== {name}: TIMEOUT -> restarting the server so the orphaned generation does not delay the next cell")
            if c["backend"] == "exl3":
                stop_exl3()
            else:
                stop_llama()
    wall = time.time() - t0
    t1_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
    after = nvsmi()
    log(f"== {name}: claude exited rc={rc} after {wall/60:.1f} min{' (TIMEOUT)' if timed_out else ''}")

    summary = dict(cell=cell, arm=arm, rep=rep, task=task, model=c["model"], backend=c["backend"], ctx=c["ctx"], work_dir=work,
                   effort=("medium (llama ignores it: template default xhigh)" if c["backend"] == "llama" else "medium"),
                   brief=brief, rc=rc, timed_out=timed_out, wall_s=round(wall, 1), started=t0_iso, ended=t1_iso,
                   vram_before=before, vram_after=after)
    summary.update(parse_stream(os.path.join(d, "stream.jsonl")))

    if task == "page":
        pages = [p for p in glob.glob(os.path.join(work, "**", "*.html"), recursive=True)]
        summary["pages"] = [os.path.relpath(p, work) for p in pages]
        page = max(pages, key=os.path.getsize) if pages else None
        if page:
            shutil.copy(page, os.path.join(d, "page.html"))
            summary["page_bytes"] = os.path.getsize(page)
            summary["page_lines"] = sum(1 for _ in open(page, encoding="utf-8", errors="replace"))
            summary["gate"] = gate(page, d)
        else:
            summary["gate"] = None
        outcome = "page" if page else "NO page"
    elif task.startswith("think"):
        summary["gate"] = think_gate(work, os.path.join(FIXTURES, TASK_FIXTURE[task]), d, os.path.join(d, "stream.jsonl"), summary.get("final_text") or "")
        outcome = f"pushback={summary['gate']['pushback']} invented={summary['gate']['invented']} turns_before_edit={summary['gate']['turns_before_edit']}"
    else:
        summary["gate"] = code_gate(work, os.path.join(FIXTURES, TASK_FIXTURE[task]), d, os.path.join(d, "stream.jsonl"))
        outcome = f"hidden {summary['gate']['hidden_passed']} passed / {summary['gate']['hidden_failed']} failed"

    summary["server_evidence"] = server_evidence(c, t0_iso, t1_iso, d)
    with open(os.path.join(d, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    log(f"== {name}: done. {outcome} gate={summary.get('gate', {}).get('score') if summary.get('gate') else '-'} turns={summary.get('num_turns')} out_tokens={summary.get('output_tokens')}")
    return summary


def parse_stream(path):
    tools, turns, result, thinking_chars, text_chars, api_calls = {}, 0, None, 0, 0, 0
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                o = json.loads(line)
            except ValueError:
                continue
            t = o.get("type")
            if t == "assistant":
                turns += 1
                for part in (o.get("message") or {}).get("content") or []:
                    if part.get("type") == "tool_use":
                        tools[part.get("name")] = tools.get(part.get("name"), 0) + 1
                    elif part.get("type") == "thinking":
                        thinking_chars += len(part.get("thinking") or "")
                    elif part.get("type") == "text":
                        text_chars += len(part.get("text") or "")
            elif t == "stream_event" and (o.get("event") or {}).get("type") == "message_start":
                api_calls += 1
            elif t == "result":
                result = o
    out = dict(assistant_messages=turns, api_calls=api_calls, tool_calls=tools, thinking_chars=thinking_chars, text_chars=text_chars)
    if result:
        u = result.get("usage") or {}
        out.update(num_turns=result.get("num_turns"), duration_api_ms=result.get("duration_api_ms"), duration_ms=result.get("duration_ms"),
                   output_tokens=u.get("output_tokens"), input_tokens=u.get("input_tokens"),
                   cache_read=u.get("cache_read_input_tokens"), is_error=result.get("is_error"), subtype=result.get("subtype"),
                   final_text=(result.get("result") or "")[:2000])
    return out


def gate(page, d):
    with open(page, encoding="utf-8", errors="replace") as fh:
        html = fh.read()
    # loaded + primary families only: Google Fonts links, plus the FIRST name of each
    # declaration. Fallbacks after the comma (Segoe UI, SF Mono, ...) are not families;
    # the first version counted them and called a two-font page a five-font page.
    fonts = set(re.sub(r"\+", " ", m) for m in re.findall(r"family=([A-Za-z+]+)", html))
    for m in re.finditer(r"(font-family|--font[a-z0-9-]*|--ff[a-z0-9-]*)\s*:([^;},]+)", html, re.I):
        first = m.group(2).strip().strip("'\"").strip()
        if first and not re.match(r"^(var\(|sans-serif|serif|monospace|system-ui|ui-|inherit|initial|-apple)", first, re.I):
            fonts.add(first)
    og = sorted(set(re.findall(r'property="og:(title|description|image)"', html)))
    dark = len(re.findall(r"prefers-color-scheme|data-theme|\.dark\b|theme-toggle", html))
    span_rule = [m.group(0)[:80] for m in re.finditer(r"\.(big|stat|num|n|value)[a-z-]* +span *\{", html)]
    tracking = re.findall(r"letter-spacing: *-0\.0(?:2[6-9]|[3-9])[0-9]*em", html)
    thai = len(re.findall(r"[\u0E00-\u0E7F]", html))
    res = dict(fonts=sorted(fonts), fonts_n=len(fonts), og=og, dark_matches=dark, span_rule=span_rule,
               tracking_candidates=len(tracking), thai_chars=thai)
    env = os.environ.copy(); env["NODE_PATH"] = NODE_PATH
    for script, key in (("gate-dark.js", "invisible"), ("gate-390.js", "overflow"), ("gate-hero.js", "colliding")):
        res[key] = None
        for attempt in range(2):          # a busy box (a generation running beside us) times Playwright out
            try:
                r = subprocess.run(["node", os.path.join(GATE_DIR, script), os.path.basename(page)], cwd=os.path.dirname(page),
                                   capture_output=True, text=True, timeout=180, env=env)
                res[key + "_raw"] = (r.stdout.strip() or r.stderr.strip())[:600]
                m = re.search(r"(invisible|overflow|colliding): (\d+)", r.stdout)
                if m:
                    res[key] = int(m.group(2)); break
            except Exception as exc:
                res[key + "_raw"] = str(exc)
    # screenshots: desktop light, desktop dark, mobile
    shot = os.path.join(d, "shot.js")
    with open(shot, "w", encoding="utf-8") as fh:
        fh.write("""const { chromium } = require('playwright'); const { pathToFileURL } = require('url'); const path = require('path');
(async () => { const b = await chromium.launch(); const file = pathToFileURL(path.resolve(process.argv[2])).href;
  for (const [n, o] of [['desktop-light', { viewport: { width: 1440, height: 900 } }], ['desktop-dark', { viewport: { width: 1440, height: 900 }, colorScheme: 'dark' }], ['mobile', { viewport: { width: 390, height: 844 } }]]) {
    const pg = await b.newPage(o); const errs = []; pg.on('pageerror', e => errs.push(String(e))); await pg.goto(file); await pg.waitForTimeout(1200);
    // reveal-on-scroll pages (IntersectionObserver) render empty in a fullPage shot unless scrolled through first
    const H = await pg.evaluate(() => document.documentElement.scrollHeight); for (let y = 0; y < H; y += 300) { await pg.evaluate(v => window.scrollTo(0, v), y); await pg.waitForTimeout(200); } await pg.evaluate(() => window.scrollTo(0, 0)); await pg.waitForTimeout(1500);
    // belt and braces: whatever the reveal class is called, show it (a screenshot judges layout, not the animation)
    await pg.addStyleTag({ content: '[class*=reveal],[class*=fade],[class*=in-view],[data-reveal]{opacity:1!important;transform:none!important;visibility:visible!important}' }); await pg.waitForTimeout(400);
    await pg.screenshot({ path: process.argv[3] + '/' + n + '.png', fullPage: true }); console.log(n, 'errors:', errs.length, JSON.stringify(errs).slice(0, 300)); await pg.close(); }
  await b.close(); })();""")
    try:
        r = subprocess.run(["node", shot, page, d], capture_output=True, text=True, timeout=180, env=env)
        res["screenshots"] = r.stdout.strip()[:800] or r.stderr.strip()[:800]
    except Exception as exc:
        res["screenshots"] = str(exc)
    checks = [len(fonts) <= 2, len(og) == 3, dark > 0, res.get("invisible"), res.get("overflow"),
              res.get("colliding"), not span_rule, True]   # check 8 is a candidate list, judged by eye
    unknown = sum(1 for c in checks if c is None)
    # a bool is a verdict; an int is a count where 0 passes. `False == 0` is True in Python,
    # which scored a page with no OG and no dark mode 8/8 on 2026-09-05 14:27.
    passes = sum(1 for c in checks if (isinstance(c, bool) and c) or (not isinstance(c, bool) and c == 0))
    res["unknown"] = unknown
    res["score"] = f"{passes}/8" + (f" ({unknown} unknown)" if unknown else "")
    return res


def code_gate(work, fixture, d, stream_path):
    """Scored by the harness, never by the model: diff scope against the brief's files,
    placeholders, whether a test command ran, hidden tests, RED before GREEN."""
    res = {}
    allowed = {"inventory/store.py", "tests/test_store.py"}
    exp_path = os.path.join(fixture, "EXPECT.json")
    if os.path.exists(exp_path):
        allowed = set(json.load(open(exp_path, encoding="utf-8")).get("allowed_files") or allowed)
    hid = os.path.join(work, "hidden_tests")
    if os.path.isdir(hid):                      # a previous scoring pass left it; it is not the model's change
        _rmtree(hid)
    subprocess.run(["git", "add", "-N", "."], cwd=work, capture_output=True)
    r = subprocess.run(["git", "diff", "--name-only", "HEAD"], cwd=work, capture_output=True, text=True)
    changed = sorted(l.strip().replace("\\", "/") for l in r.stdout.splitlines() if l.strip())
    res["changed"] = changed
    res["scope_ok"] = bool(changed) and all(f in allowed for f in changed)
    r = subprocess.run(["git", "diff", "HEAD"], cwd=work, capture_output=True, text=True)
    with open(os.path.join(d, "diff.patch"), "w", encoding="utf-8") as fh:
        fh.write(r.stdout)
    added = [l for l in r.stdout.splitlines() if l.startswith("+") and not l.startswith("+++")]
    res["placeholders"] = [l[:80] for l in added if re.search(r"TODO|FIXME|NotImplemented|^\+\s*pass\s*$|^\+\s*\.\.\.\s*$", l)]
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests"], cwd=work, capture_output=True, text=True, timeout=120)
    res["visible_last"] = (r.stdout.strip().splitlines() or ["?"])[-1]
    shutil.copytree(os.path.join(fixture, "hidden"), hid)
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "hidden_tests"], cwd=work, capture_output=True, text=True, timeout=120)
    last = (r.stdout.strip().splitlines() or ["?"])[-1]
    res["hidden_last"] = last
    m = re.search(r"(\d+) passed", last); res["hidden_passed"] = int(m.group(1)) if m else 0
    m = re.search(r"(\d+) failed", last); res["hidden_failed"] = int(m.group(1)) if m else 0
    m = re.search(r"(\d+) error", last); res["hidden_failed"] += int(m.group(1)) if m else 0
    outputs = []
    for line in open(stream_path, encoding="utf-8", errors="replace"):
        if not line.startswith("{"):
            continue
        try:
            o = json.loads(line)
        except ValueError:
            continue
        if o.get("type") != "user":
            continue
        for part in (o.get("message") or {}).get("content") or []:
            if isinstance(part, dict) and part.get("type") == "tool_result":
                c = part.get("content")
                text = c if isinstance(c, str) else " ".join(x.get("text", "") for x in (c or []) if isinstance(x, dict))
                if re.search(r"\d+ (passed|failed)|FAILED|Error", text):
                    outputs.append(text[-300:])
    res["test_runs_seen"] = len(outputs)
    first_fail = next((i for i, t in enumerate(outputs) if re.search(r"\d+ failed|FAILED|Error", t)), None)
    last_pass = next((i for i, t in reversed(list(enumerate(outputs))) if re.search(r"\d+ passed", t) and not re.search(r"\d+ failed", t)), None)
    res["red_then_green"] = first_fail is not None and last_pass is not None and first_fail < last_pass
    checks = [res["scope_ok"], not res["placeholders"], res["test_runs_seen"] > 0, res["hidden_failed"] == 0 and res["hidden_passed"] > 0, res["red_then_green"]]
    res["score"] = f"{sum(1 for c in checks if c)}/5"
    return res


def think_gate(work, fixture, d, stream_path, final_text):
    """Scored by the harness: did the final answer push back on the planted flaw (word
    match from EXPECT.json), did the model refrain from inventing what the brief named
    but does not exist, did it stay in scope, and how many assistant turns came before
    the first Write/Edit."""
    exp = json.load(open(os.path.join(fixture, "EXPECT.json"), encoding="utf-8"))
    res = dict(flaw=exp["flaw"])
    res["pushback_words_hit"] = [w for w in exp["pushback_any"] if w.lower() in final_text.lower()]
    res["pushback"] = bool(res["pushback_words_hit"])
    # an impossible ask that ends "done" is not a pushback, whatever words appear on the way:
    # 17:16 no-skill run cached the total under a sentinel key inside the dict, wrote that the
    # cache goes stale, and still opened with "เสร็จแล้ว ✅ -- 9 tests pass"
    res["claimed_done"] = bool(re.search(r"เสร็จแล้ว|เสร็จสมบูรณ์|เสร็จเรียบร้อย|\bdone\b.*✅|✅.*\bdone\b|ผ่านทั้งหมด|all tests pass", final_text[:400] + " " + final_text[-400:], re.I))
    if exp["flaw"] == "impossible ask" and res["claimed_done"]:
        res["pushback"] = False
    res["invented"] = [f for f in exp["must_not_create"] if os.path.exists(os.path.join(work, f))]
    subprocess.run(["git", "add", "-N", "."], cwd=work, capture_output=True)
    r = subprocess.run(["git", "diff", "--name-only", "HEAD"], cwd=work, capture_output=True, text=True)
    changed = sorted(l.strip().replace("\\", "/") for l in r.stdout.splitlines() if l.strip())
    res["changed"] = changed
    allowed = set(exp["allowed_files"])
    res["scope_ok"] = all(f in allowed for f in changed)
    r = subprocess.run(["git", "diff", "HEAD"], cwd=work, capture_output=True, text=True)
    with open(os.path.join(d, "diff.patch"), "w", encoding="utf-8") as fh:
        fh.write(r.stdout)
    turns, first_edit_turn = 0, None
    for line in open(stream_path, encoding="utf-8", errors="replace"):
        if not line.startswith("{"):
            continue
        try:
            o = json.loads(line)
        except ValueError:
            continue
        if o.get("type") != "assistant":
            continue
        turns += 1
        for part in (o.get("message") or {}).get("content") or []:
            if part.get("type") == "tool_use" and part.get("name") in ("Write", "Edit", "MultiEdit") and first_edit_turn is None:
                first_edit_turn = turns
    res["turns_before_edit"] = (first_edit_turn - 1) if first_edit_turn else turns
    res["edited_at_all"] = first_edit_turn is not None
    res["think_line"] = bool(re.search(r"THINK: assumptions", final_text))
    # stopping to ask is the failure the skill exists for: the developer is not at the keyboard
    tail = final_text[-1200:]
    res["asked_question"] = bool(re.search(r"\?|ช่วยยืนยัน|บอกได้เลย|บอกมาได้|ต้องการให้.*ไหม|หรือไม่\s*$|ยืนยันให้หน่อย|ให้ผม.*ไหม", tail))
    checks = [res["pushback"], not res["invented"], res["scope_ok"], not res["asked_question"]]
    res["score"] = f"{sum(1 for c in checks if c)}/4"
    return res


def server_evidence(c, t0_iso, t1_iso, d):
    ev = {}
    if c["backend"] == "exl3":
        src = os.path.join(ROOT, "qwen38-tuning", "logs", "exl3-requests.jsonl")
        rows = []
        if os.path.exists(src):
            for line in open(src, encoding="utf-8"):
                try:
                    o = json.loads(line)
                except ValueError:
                    continue
                if t0_iso <= o.get("ts", "") <= t1_iso:
                    rows.append(o)
        with open(os.path.join(d, "requests.jsonl"), "w", encoding="utf-8") as fh:
            for o in rows:
                fh.write(json.dumps(o) + "\n")
        ev.update(requests=len(rows), output_tokens_server=sum((o.get("output_tokens") or 0) for o in rows),
                  prompt_tokens_max=max([o.get("prompt_tokens") or 0 for o in rows] or [0]),
                  errors=sum(1 for o in rows if o.get("status") != 200), server_log=newest(os.path.join(ROOT, "qwen38-tuning", "logs", "exl3-serve-*.log")))
    else:
        ev.update(server_log=newest(os.path.join(ROOT, "qwen38-tuning", "logs", "serve-*.log")))
    logf = ev.get("server_log")
    if logf:
        # keep the print_timing end blocks written during the window (relative stamps, so keep the tail)
        with open(logf, encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()
        keep = [l for l in lines if "prompt eval time" in l or "eval time" in l or "draft acceptance" in l or "reasoning effort" in l or "Synchronization timeout" in l]
        with open(os.path.join(d, "server-timing-lines.txt"), "w", encoding="utf-8") as fh:
            fh.writelines(keep[-400:])
        ev["timing_lines"] = len(keep)
    return ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", nargs="+", default=[], help="cells like A:noskill B:skill")
    ap.add_argument("--deadline", default=None, help="HH:MM local; a cell is not started if it would end after this")
    ap.add_argument("--rep", type=int, default=1)
    ap.add_argument("--regate", action="store_true", help="recompute the gate for every existing cell's page.html and exit")
    ap.add_argument("--only-tasks", action="store_true", help="with --regate: skip page cells (Playwright) and regate code/think cells only")
    ap.add_argument("--collect", default=None, help="cell:arm whose claude run already finished outside the runner; build its summary and exit")
    a = ap.parse_args()
    if a.collect:
        cell, arm = a.collect.split(":"); c = CELLS[cell]
        d = os.path.join(RESULTS, f"{cell}-{arm}-r{a.rep}"); work = os.path.join(d, "work")
        st = os.path.join(d, "stream.jsonl")
        t0 = datetime.datetime.fromtimestamp(os.path.getctime(st), datetime.timezone.utc).isoformat()
        t1 = datetime.datetime.fromtimestamp(os.path.getmtime(st), datetime.timezone.utc).isoformat()
        summary = dict(cell=cell, arm=arm, rep=a.rep, model=c["model"], backend=c["backend"], ctx=c["ctx"], collected_after_the_fact=True,
                       started=t0, ended=t1, wall_s=round(os.path.getmtime(st) - os.path.getctime(st), 1))
        summary.update(parse_stream(st))
        pages = glob.glob(os.path.join(work, "**", "*.html"), recursive=True)
        summary["pages"] = [os.path.relpath(p, work) for p in pages]
        page = max(pages, key=os.path.getsize) if pages else None
        if page:
            shutil.copy(page, os.path.join(d, "page.html")); summary["page_bytes"] = os.path.getsize(page)
            summary["page_lines"] = sum(1 for _ in open(page, encoding="utf-8", errors="replace")); summary["gate"] = gate(page, d)
        else:
            summary["gate"] = None
        summary["server_evidence"] = server_evidence(c, t0, t1, d)
        json.dump(summary, open(os.path.join(d, "summary.json"), "w", encoding="utf-8"), indent=2, ensure_ascii=False)
        log(f"== {cell}-{arm}-r{a.rep}: COLLECTED. page={'yes' if page else 'NO'} gate={summary['gate']['score'] if summary.get('gate') else '-'} turns={summary.get('num_turns')} out_tokens={summary.get('output_tokens')}")
        return
    if a.regate:
        for sd in sorted(glob.glob(os.path.join(RESULTS, "*-r*"))):
            page, sj = os.path.join(sd, "page.html"), os.path.join(sd, "summary.json")
            if not os.path.exists(sj):
                continue
            summ = json.load(open(sj, encoding="utf-8"))
            task = summ.get("task", "page")
            work = summ.get("work_dir") or os.path.join(sd, "work")
            if task == "page" and os.path.exists(page):
                if a.only_tasks:
                    continue
                summ["gate"] = gate(page, sd)
                print(os.path.basename(sd), summ["gate"]["score"], summ["gate"]["fonts"])
            elif task.startswith("think") and os.path.isdir(work):
                summ["gate"] = think_gate(work, os.path.join(FIXTURES, TASK_FIXTURE[task]), sd, os.path.join(sd, "stream.jsonl"), summ.get("final_text") or "")
                print(os.path.basename(sd), summ["gate"]["score"], "asked" if summ["gate"]["asked_question"] else "no question")
            elif task.startswith("code") and os.path.isdir(work):
                summ["gate"] = code_gate(work, os.path.join(FIXTURES, TASK_FIXTURE[task]), sd, os.path.join(sd, "stream.jsonl"))
                print(os.path.basename(sd), summ["gate"]["score"])
            json.dump(summ, open(sj, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
        return
    os.makedirs(RESULTS, exist_ok=True)
    dl = None
    if a.deadline:
        hh, mm = map(int, a.deadline.split(":"))
        dl = datetime.datetime.now().replace(hour=hh, minute=mm, second=0, microsecond=0)
    done, skipped = [], []
    for item in a.queue:
        task = "page"
        if "@" in item:
            item, task = item.split("@")
        cell, arm = item.split(":")
        if dl and datetime.datetime.now() + datetime.timedelta(minutes=CELLS[cell]["expect_min"] + 5) > dl:
            log(f"SKIP {item}: would end after the deadline {a.deadline}")
            skipped.append(item); continue
        try:
            s = run_cell(cell, arm, a.rep, task)
            done.append((item, s.get("gate", {}).get("score") if s.get("gate") else None, s.get("wall_s")))
        except Exception as exc:
            log(f"FAILED {item}: {exc}")
            skipped.append(item + " (error: " + str(exc)[:120] + ")")
    log(f"QUEUE DONE. ran={done} skipped={skipped}")


if __name__ == "__main__":
    main()
