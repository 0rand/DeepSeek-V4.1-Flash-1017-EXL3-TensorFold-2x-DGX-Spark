"""hwcorr.py R0LOG R1LOG HW_HEAD HW_WORKER: each prompt chunk's window against both nodes' 0.5 s samples.
Prints, for slow chunks (> SLOW s) and normal ones: mean GPU clock/power/temp/util, throttle-reason hits,
host CPU busy and MHz, and the rate of reclaim, compaction, major faults, PSI and context switches."""
import datetime as dt
import re
import statistics as st
import sys

SLOW = 1.3
LOCAL = dt.timezone(dt.timedelta(hours=-7))      # nvidia-smi prints local time (UTC-7)


def chunks(path):
    out = []
    for line in open(path, errors="replace"):
        m = re.match(r"(\S+)Z \[tensorfold\] rank (\d) chunk at (\d+): ([\d.]+)s", line)
        if not m:
            continue
        end = dt.datetime.fromisoformat(m.group(1)[:26]).replace(tzinfo=dt.timezone.utc).timestamp()
        d = float(m.group(4))
        out.append((end - d, end, int(m.group(3)), d))
    return out


def gpu(path):
    rows = []
    for line in open(path, errors="replace"):
        p = [x.strip() for x in line.split(",")]
        if len(p) < 8 or "MHz" not in p[1]:
            continue
        t = dt.datetime.strptime(p[0], "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=LOCAL).timestamp()
        num = lambda s: float(re.sub(r"[^\d.]", "", s) or 0)
        rows.append((t, {"sm_mhz": num(p[1]), "power_w": num(p[3]), "temp_c": num(p[4]),
                         "throttle": int(p[5], 16) != 0, "gpu_util": num(p[6])}))
    return rows


def host(path):
    rows = []
    for line in open(path, errors="replace"):
        p = line.split()
        if len(p) < 4:
            continue
        kv = dict(x.split("=", 1) for x in p[1:] if "=" in x)
        r = {"busy": float(kv["busy"]), "cpu_mhz": float(kv.get("mhz") or 0), "free_mb": float(kv["free"])}
        for k in ("pgscan_kswapd", "pgscan_direct", "compact_stall", "pgmajfault", "pswpout", "ctxt", "pgmigrate_success", "compact_daemon_migrate_scanned", "compact_migrate_scanned", "thp_fault_alloc"):
            r["c_" + k] = float(kv.get(k, 0))
        for k in ("memory", "io", "cpu"):
            r["psi_" + k] = float(kv.get(k, "0/0").split("/")[1])
        rows.append((float(p[0]), r))
    return rows


def window(rows, a, b, rate_keys=()):
    inside = [r for t, r in rows if a <= t <= b]
    if not inside:
        return {}
    res = {}
    for k in inside[0]:
        if k.startswith("c_") or k.startswith("psi_"):
            before = [r for t, r in rows if t <= a]
            after = [r for t, r in rows if t >= b]
            if before and after:
                res[k + "/s"] = (after[0][k] - before[-1][k]) / max(b - a, 0.5)
        elif k == "throttle":
            res[k] = sum(1 for r in inside if r[k])
        else:
            res[k] = st.mean(r[k] for r in inside)
    return res


def main():
    r0, r1, hh, hw = sys.argv[1:5]
    ch = [c for c in chunks(r0) if c[3] > 0.3]                 # prompt chunks of 2048 rows (not tiny decode prompts)
    nodes = {"head": (gpu(hh + "/gpu.csv"), host(hh + "/host.txt")),
             "worker": (gpu(hw + "/gpu.csv"), host(hw + "/host.txt"))}
    t_lo = min(min(t for t, _ in g) for g, _ in nodes.values())
    t_hi = max(max(t for t, _ in g) for g, _ in nodes.values())
    ch = [c for c in ch if t_lo <= c[0] and c[1] <= t_hi]
    slow = [c for c in ch if c[3] > SLOW]
    norm = [c for c in ch if c[3] <= SLOW]
    print(f"chunks in the sample window: {len(ch)}, slow (> {SLOW} s): {len(slow)}")
    for name, (g, h) in nodes.items():
        print(f"\n== {name}")
        keys = None
        agg = {}
        for label, group in (("normal", norm), ("slow", slow)):
            vals = [dict(window(g, a, b), **window(h, a, b)) for a, b, _, _ in group]
            vals = [v for v in vals if v]
            if not vals:
                continue
            keys = keys or sorted(vals[0])
            agg[label] = {k: st.mean(v.get(k, 0) for v in vals) for k in keys}
        for k in keys or []:
            n, s = agg.get("normal", {}).get(k, 0), agg.get("slow", {}).get(k, 0)
            print(f"  {k:22s} normal {n:12.1f}   slow {s:12.1f}")
    print("\nslowest chunks (start UTC, position, seconds) with head/worker GPU clock and power:")
    for a, b, pos, d in sorted(slow, key=lambda c: -c[3])[:12]:
        hg, wg = window(nodes["head"][0], a, b), window(nodes["worker"][0], a, b)
        hh_, wh = window(nodes["head"][1], a, b), window(nodes["worker"][1], a, b)
        print(f"  {dt.datetime.fromtimestamp(a, dt.timezone.utc):%H:%M:%S} pos {pos:6d} {d:.2f}s | head "
              f"{hg.get('sm_mhz', 0):.0f} MHz {hg.get('power_w', 0):.0f} W cpu {hh_.get('busy', 0):.0f}% "
              f"scan {hh_.get('c_pgscan_kswapd/s', 0) + hh_.get('c_pgscan_direct/s', 0):.0f}/s | worker "
              f"{wg.get('sm_mhz', 0):.0f} MHz {wg.get('power_w', 0):.0f} W cpu {wh.get('busy', 0):.0f}% "
              f"scan {wh.get('c_pgscan_kswapd/s', 0) + wh.get('c_pgscan_direct/s', 0):.0f}/s")


main()
