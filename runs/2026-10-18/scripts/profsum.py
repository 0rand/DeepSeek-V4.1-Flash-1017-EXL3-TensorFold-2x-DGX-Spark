"""profsum.py DIR [DIR...]: summary of profbench.sh runs.
Per run: prefill tok/s per prompt length (each pass), slow chunks (> SLOW s) on each rank, direct reclaim and
compaction stalls of the engine (count, total ms, how many fall inside slow chunks), and the CPU type under the
busiest engine thread (A725 slow cores 0-4,10-14; X925 fast cores 5-9,15-19) for slow and normal chunks."""
import datetime as dt
import glob
import json
import os
import re
import statistics as st
import sys

SLOW = 1.3
SLOWCPU = set(range(0, 5)) | set(range(10, 15))


def chunks(path):
    out = []
    for line in open(path, errors="replace"):
        m = re.match(r"(\S+)Z \[tensorfold\] rank (\d) chunk at (\d+): ([\d.]+)s", line)
        if m:
            end = dt.datetime.fromisoformat(m.group(1)[:26]).replace(tzinfo=dt.timezone.utc).timestamp()
            d = float(m.group(4))
            if int(m.group(3)) > 0 or d > 0.5:     # skip the tiny decode-side "chunk at 0" lines
                out.append((end - d, end, d))
    return out


def stalls(path, pids):
    ev = []
    for line in open(path, errors="replace"):
        m = re.match(r"([\d.]+) (DR|CP) (\S+) pid=(\d+) tid=(\d+) cpu=(\d+) us=(\d+)", line)
        if m and int(m.group(4)) in pids:
            us = int(m.group(7))
            ev.append((float(m.group(1)) - us / 1e6, float(m.group(1)), m.group(2), us))
    return ev


def threads(path):
    rows = []
    for line in open(path, errors="replace"):
        p = line.split()
        if len(p) >= 2:
            rows.append((float(p[0]), [tuple(int(x) for x in q.split(":")) for q in p[1:] if q.count(":") == 2]))
    return rows


def busiest_cpu(rows, a, b):
    """CPU of the thread that gained the most CPU ticks in [a, b] (sampled at each 0.5 s point)."""
    inside = [r for r in rows if a <= r[0] <= b]
    if len(inside) < 2:
        return []
    gain = {}
    for t in inside[0][1]:
        gain[t[0]] = 0
    first = {t[0]: t[2] for t in inside[0][1]}
    last = {t[0]: t[2] for t in inside[-1][1]}
    top = max(last, key=lambda k: last[k] - first.get(k, last[k]))
    return [next(t[1] for t in r[1] if t[0] == top) for r in inside if any(t[0] == top for t in r[1])]


def engine_pids(d, node):
    hot = os.path.join(d, node, "hot-tids")
    return {int(x) for x in open(hot).read().split()} if os.path.exists(hot) else set()


def run(d):
    print(f"## {os.path.basename(d.rstrip('/'))}")
    for f in sorted(glob.glob(os.path.join(d, "pf-*-long.json")) + glob.glob(os.path.join(d, "pf-*-8k.json"))):
        res = json.load(open(f))["result"]
        print("  ", os.path.basename(f), " ".join(f"{r['length']//1024}K={r['prefill_tps']:.0f}" for r in res))
    for rank, node in ((0, "head"), (1, "worker")):
        ch = chunks(os.path.join(d, f"r{rank}.log"))
        slow = [c for c in ch if c[2] > SLOW]
        tr = threads(os.path.join(d, node, "threads.txt"))
        mem = os.path.join(d, node, "mem.txt")
        allpid = set()
        for line in open(mem, errors="replace"):
            m = re.search(r" (DR|CP) (\S+) pid=(\d+)", line)
            if m and m.group(2) == "tensorfold":
                allpid.add(int(m.group(3)))
        ev = stalls(mem, allpid)
        inslow = [e for e in ev if any(c[0] <= e[1] and e[0] <= c[1] for c in slow)]
        cpu_slow = [x for c in slow for x in busiest_cpu(tr, c[0], c[1])]
        cpu_norm = [x for c in ch if c[2] <= SLOW for x in busiest_cpu(tr, c[0], c[1])]
        frac = lambda xs: f"{100 * sum(x in SLOWCPU for x in xs) / len(xs):.0f}%" if xs else "n/a"
        med = st.median([c[2] for c in ch]) if ch else 0
        print(f"   rank {rank}: {len(ch)} chunks, median {med:.2f}s, slow {len(slow)} "
              f"({', '.join(f'{c[2]:.2f}' for c in sorted(slow, key=lambda c: -c[2])[:6])})")
        for k in ("DR", "CP"):
            e = [x for x in ev if x[2] == k]
            s = [x for x in inslow if x[2] == k]
            print(f"     {k}: {len(e)} events, {sum(x[3] for x in e)/1000:.0f} ms total, max {max([x[3] for x in e] or [0])/1000:.1f} ms;"
                  f" in slow chunks {len(s)} ({sum(x[3] for x in s)/1000:.0f} ms)")
        print(f"     busiest thread on an A725 core: slow chunks {frac(cpu_slow)}, normal chunks {frac(cpu_norm)}")


for d in sys.argv[1:]:
    run(d)
