"""llama-server-shaped console timing for the EXL3 server (issue #71, 2026-09-04).

Two things, both ours and both outside the fork's file:

* `LiveTiming` — one line every PERIOD seconds while a request runs, in the
  shape of llama-server's `slot print_timing` lines (prompt processing with
  pp / pp_3s, then n_gen with tg / tg_3s), stamped with elapsed time since
  the server started.
* `report()` — the end-of-request block (`prompt eval time`, `eval time`,
  `total time`, `draft acceptance`, `reasoning effort`) and the llama.cpp-style
  `timings` dict that goes into the response.

The fork's Job accumulates time_prefill = first_token - first_prefill and
time_generate = last_token - first_token (generator/job.py:674-675), two
disjoint intervals, so the decode rate is out_toks / time_generate
(CORRECTIONS.md §47, instrument fault 14). cached_tokens are prompt tokens
served from the page cache that did not run through prefill.
"""
import time


class LiveTiming:
    PERIOD = 3.0
    _task = 0
    _t_start = time.time()

    @classmethod
    def stamp(cls):
        e = time.time() - cls._t_start
        m, rem = divmod(e, 60.0)
        return f"{int(m)}.{int(rem):02d}.{int((rem - int(rem)) * 1000):03d} I"

    @classmethod
    def prefix(cls):
        return f"{cls.stamp()} slot print_timing: id  0 | task {cls._task} |"

    def __init__(self, n_prompt):
        LiveTiming._task += 1
        self.task = LiveTiming._task
        self.n_prompt = n_prompt
        self.t0 = time.time()
        self.pp_hist = []          # (t, tokens prefilled)
        self.pp_last_print = 0.0
        self.t_gen0 = None
        self.n_gen = 0
        self.tg_hist = []          # (t, tokens generated)
        self.tg_last_print = None

    @staticmethod
    def rate3(hist, now):
        """Tokens per second over the trailing three seconds of `hist`."""
        cutoff = now - 3.0
        old = hist[0]
        for h in hist:
            if h[0] >= cutoff:
                break
            old = h
        dt = now - old[0]
        return (hist[-1][1] - old[1]) / dt if dt > 0.2 else 0.0

    def prefill(self, curr):
        now = time.time()
        self.pp_hist.append((now, curr))
        if now - self.pp_last_print < self.PERIOD and curr < self.n_prompt:
            return
        self.pp_last_print = now
        dt = now - self.t0
        pp = curr / dt if dt > 0.05 else 0.0
        print(f"{self.stamp()} slot print_timing: id  0 | task {self.task} | prompt processing, "
              f"n_tokens = {curr:>6d}, progress = {curr / max(self.n_prompt, 1):.2f}, t = {dt:6.2f} s / "
              f"{pp:.2f} tokens per second, pp_3s = {self.rate3(self.pp_hist, now):.2f}", flush = True)

    def generated(self, n_new):
        now = time.time()
        if self.t_gen0 is None:
            self.t_gen0 = now
        self.n_gen += n_new
        self.tg_hist.append((now, self.n_gen))
        if self.tg_last_print is None:
            self.tg_last_print = now
            return
        if now - self.tg_last_print < self.PERIOD:
            return
        self.tg_last_print = now
        dt = now - self.t_gen0
        tg = self.n_gen / dt if dt > 0.05 else 0.0
        print(f"{self.stamp()} slot print_timing: id  0 | task {self.task} | n_gen = {self.n_gen:>6d}, "
              f"tg = {tg:6.2f} t/s, tg_3s = {self.rate3(self.tg_hist, now):6.2f} t/s", flush = True)


def timings(final_res, prompt_toks, out_toks, wall):
    """The llama.cpp-style `timings` dict for a finished request."""
    tp = float(final_res.get("time_prefill") or 0.0)
    tg = float(final_res.get("time_generate") or 0.0)
    cached = int(final_res.get("cached_tokens") or 0)
    pn = max(prompt_toks - cached, 0)
    return {
        "prompt_n": pn, "prompt_ms": round(tp * 1000, 1),
        "prompt_per_second": round(pn / tp, 1) if tp > 0 else None,
        "predicted_n": out_toks, "predicted_ms": round(tg * 1000, 1),
        "predicted_per_second": round(out_toks / tg, 1) if tg > 0 else None,
        "cached_tokens": cached, "wall_ms": round(wall * 1000, 1),
        "draft_accepted": final_res.get("accepted_draft_tokens"),
        "draft_rejected": final_res.get("rejected_draft_tokens"),
    }


def report(final_res, prompt_toks, out_toks, wall, effort, out = print):
    """Print the end-of-request block and return the `timings` dict."""
    t = timings(final_res, prompt_toks, out_toks, wall)
    pn, cached = t["prompt_n"], t["cached_tokens"]
    tp, tg = t["prompt_ms"] / 1000, t["predicted_ms"] / 1000
    pps, tps = t["prompt_per_second"] or 0.0, t["predicted_per_second"] or 0.0
    acc, rej = t["draft_accepted"] or 0, t["draft_rejected"] or 0
    pre = LiveTiming.prefix()
    ms_tok_p = (tp * 1000 / pn) if pn else 0.0
    ms_tok_g = (tg * 1000 / out_toks) if out_toks else 0.0
    out(f"{pre} prompt eval time = {tp * 1000:10.2f} ms / {pn:5d} tokens ({ms_tok_p:8.2f} ms per token, {pps:8.2f} tokens per second)"
        + (f"  [{cached} cached]" if cached else ""), flush = True)
    out(f"{pre}        eval time = {tg * 1000:10.2f} ms / {out_toks:5d} tokens ({ms_tok_g:8.2f} ms per token, {tps:8.2f} tokens per second)", flush = True)
    out(f"{pre}       total time = {wall * 1000:10.2f} ms / {pn + out_toks:5d} tokens", flush = True)
    if acc or rej:
        out(f"{pre} draft acceptance = {acc / max(acc + rej, 1):.5f} ({acc:5d} accepted / {acc + rej:5d} generated)", flush = True)
    out(f"{pre} reasoning effort = {effort}", flush = True)
    return t
