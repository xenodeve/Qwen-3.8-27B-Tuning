# logs — server output and the queue timeline

480+ files. Two kinds, and you almost always want the first.

## `afk-driver.log` — the timeline

Every unattended step writes `START` / `DONE` / `FAIL` here with a timestamp.
**This is where you find out what happened while you were away.**

```sh
tail -30 afk-driver.log
grep -E "FAIL|SKIP" afk-driver.log        # what went wrong
```

A `FAIL … (rc=0)` line is a bug in an old copy of the queue's `step()` function
— `$(date)` reset `$?` before the exit code was read, so the real code was lost.
Fixed 2026-08-20; older lines cannot be diagnosed from the code alone.

## Per-step and per-boot logs

| pattern | what it is |
|---|---|
| `q38-<step>.log` | one queue step's stdout — the paired verdict tables live here |
| `depth-<arm>-<kv>-r<n>-<ctx>.log` | one `llama-server` boot at verbosity 5 |
| `arena-r<n>-<arm>.log` | one arena boot |
| `swap-<profile>.log` | output of a server started by `swap-model.sh` |

---

## What the boot logs are good for

They are the only place several facts appear, and grepping them costs no GPU
time at all. Two open questions were closed this way on 2026-08-20:

```sh
# which artifacts carry the MTP head (blk.64)
grep -oE "blk\.64\.[a-z_0-9]+" arena-r1-v3-q2kxl.log | sort -u

# what is actually inside a "1-bit" file
grep -oE "type +[a-z0-9_]+: +[0-9]+ tensors" arena-r0-iq1m-nomtp.log | sort -u

# parameter count and the loader's own bits-per-weight
grep -E "model params|file type" arena-r1-v3-iq1s.log | head -2

# why a boot failed
grep -iE "error|failed" <boot>.log | head -5
```

That last one found `quantized V cache requires flash_attn to be enabled`, which
answered whether `-fa` is a choice on this machine. It is not.

---

These files are large and disposable. The numbers that matter are extracted into
[`../results/`](../results/); the conclusions into
[`../../docs/reports/`](../../docs/reports/).
