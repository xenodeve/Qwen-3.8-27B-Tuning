"""Live tok/s for the EXL3 OpenAI server, like llama-server's terminal line.

Polls GET /health (cumulative prompt/completion token counters the fork bumps
per prefill chunk and per generated token) four times a second and prints one
refreshing line:

    pp 12,480 tok  412.3 tok/s  pp3 398.1 | tg 96 tok  52.7 tok/s  tg3 54.1 | busy 7.2 s

  pp / tg  : this request's cumulative prefill / decode rate
  pp3 / tg3: rate over the trailing 3 seconds only
When the server goes idle the line is frozen with a `done` summary and a new
line starts for the next request. Works against the server as it runs; no
server change or restart needed.

    python exl3-live.py [--url http://127.0.0.1:8000] [--window 3] [--interval 0.25]
"""
import argparse, collections, json, sys, time, urllib.request


def fetch(url):
    with urllib.request.urlopen(url + "/health", timeout=2) as r:
        return json.loads(r.read())


def rate(samples, key, window, now):
    """Tokens per second of `key` over the trailing `window` seconds."""
    cutoff = now - window
    old = None
    for t, s in samples:
        if t >= cutoff:
            break
        old = (t, s)
    if old is None:
        old = samples[0]
    t0, s0 = old
    t1, s1 = samples[-1]
    dt = t1 - t0
    return (s1[key] - s0[key]) / dt if dt > 0.2 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--window", type=float, default=3.0)
    ap.add_argument("--interval", type=float, default=0.25)
    a = ap.parse_args()

    samples = collections.deque(maxlen=int(60 / a.interval))
    req = None            # state of the request in flight
    last_line = ""
    try:
        while True:
            now = time.time()
            try:
                h = fetch(a.url)
            except Exception as e:
                sys.stdout.write(f"\r{'':100}\r[server unreachable: {e}]")
                sys.stdout.flush()
                time.sleep(1.0)
                continue
            s = {"pp": h["prompt_tokens_total"], "tg": h["completion_tokens_total"]}
            samples.append((now, s))
            busy = bool(h.get("busy"))

            if busy and req is None:
                req = {"t0": now, "pp0": s["pp"], "tg0": s["tg"], "t_first_tg": None, "t_pp_end": None}
            if req is not None:
                pp_n = s["pp"] - req["pp0"]
                tg_n = s["tg"] - req["tg0"]
                if tg_n > 0 and req["t_first_tg"] is None:
                    req["t_first_tg"] = now
                    req["t_pp_end"] = now
                pp_dt = (req["t_pp_end"] or now) - req["t0"]
                tg_dt = (now - req["t_first_tg"]) if req["t_first_tg"] else 0.0
                pp_rate = pp_n / pp_dt if pp_dt > 0.2 else 0.0
                tg_rate = tg_n / tg_dt if tg_dt > 0.2 else 0.0
                line = (f"pp {pp_n:>7,} tok {pp_rate:7.1f} tok/s  pp3 {rate(samples, 'pp', a.window, now):6.1f}"
                        f" | tg {tg_n:>5,} tok {tg_rate:6.1f} tok/s  tg3 {rate(samples, 'tg', a.window, now):6.1f}"
                        f" | {'busy' if busy else 'done'} {now - req['t0']:5.1f} s")
                sys.stdout.write("\r" + line + " " * max(0, len(last_line) - len(line)))
                sys.stdout.flush()
                last_line = line
                if not busy:
                    sys.stdout.write("\n")
                    sys.stdout.flush()
                    req = None
                    last_line = ""
            else:
                idle = f"idle  (totals: pp {s['pp']:,}  tg {s['tg']:,}  ctx {h.get('context_length')})"
                sys.stdout.write("\r" + idle + " " * max(0, len(last_line) - len(idle)))
                sys.stdout.flush()
                last_line = idle
            time.sleep(a.interval)
    except KeyboardInterrupt:
        sys.stdout.write("\n")


if __name__ == "__main__":
    main()
