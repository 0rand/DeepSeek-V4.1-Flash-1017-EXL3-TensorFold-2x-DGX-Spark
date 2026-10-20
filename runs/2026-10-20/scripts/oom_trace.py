"""Memory trace for recipe issue #21: MemAvailable on both nodes every second while the engine serves
(1) 4 long prompts at the same time and (2) image-heavy requests at the same time.

Run on the head: python3 -I oom_trace.py TAG
Writes oom-TAG.csv (t, head_kib, worker_kib, phase) and oom-TAG.json (requests, minimums, earlyoom lines).
"""
import concurrent.futures as cf
import json
import random
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from val_checks import image_msg, post  # noqa: E402

TAG = sys.argv[1]
OUT = Path(__file__).parent
phase = ["idle"]
rows = []
stop = threading.Event()
WORDS = ("alpha beta gamma delta engine kernel prefill decode cache memory node spark tensor graph stream token "
         "router lane spill page fault buffer queue latency budget sample draft verify layer expert gate").split()


def head_avail():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1])
    return -1


def sampler():
    w = subprocess.Popen(["ssh", "-o", "BatchMode=yes", "10.100.80.2",
                          "while true; do awk '/MemAvailable/{print $2; fflush()}' /proc/meminfo; sleep 1; done"],
                         stdout=subprocess.PIPE, text=True)
    last_w = [-1]

    def read_worker():
        for line in w.stdout:
            try:
                last_w[0] = int(line)
            except ValueError:
                pass
    threading.Thread(target=read_worker, daemon=True).start()
    t0 = time.time()
    while not stop.is_set():
        rows.append((round(time.time() - t0, 1), head_avail(), last_w[0], phase[0]))
        time.sleep(1)
    w.kill()


def long_text(tokens, seed):
    rng = random.Random(seed)
    n = int(tokens * 0.75)                                    # about 0.75 words per token for this vocabulary
    words = [rng.choice(WORDS) + str(rng.randint(0, 99)) for _ in range(n)]
    return " ".join(words)


def long_req(tokens, seed):
    t = time.time()
    s, d = post({"messages": [{"role": "user", "content": long_text(tokens, seed)
                               + "\n\nCount how often the word 'spark' appears above. Answer briefly."}],
                 "max_tokens": 512, "chat_template_kwargs": {"enable_thinking": False}}, timeout=3600)
    u = d.get("usage", {})
    return {"kind": f"text~{tokens // 1000}K", "status": s, "seconds": round(time.time() - t, 1),
            "prompt_tokens": u.get("prompt_tokens"), "completion_tokens": u.get("completion_tokens"),
            "error": None if s == 200 else str(d)[:300]}


def image_req(n, seed):
    t = time.time()
    msgs = image_msg(n, long_text(20000, seed) + "\n\nWhat do these screenshots show? One sentence.",
                     size=(1024, 768), seed=seed * 100)
    try:
        s, d = post({"messages": msgs, "max_tokens": 256, "chat_template_kwargs": {"enable_thinking": False}},
                    timeout=3600)
    except Exception as exc:  # noqa: BLE001  the engine died: record it
        s, d = -1, {"exception": repr(exc)[:200]}
    u = d.get("usage", {})
    return {"kind": f"{n} images + ~20K text", "status": s, "seconds": round(time.time() - t, 1),
            "prompt_tokens": u.get("prompt_tokens"), "error": None if s == 200 else str(d)[:300]}


def phase_min(name):
    sel = [r for r in rows if r[3] == name]
    return {"head_min_mib": min((r[1] for r in sel), default=-1) // 1024,
            "worker_min_mib": min((r[2] for r in sel if r[2] > 0), default=-1) // 1024, "samples": len(sel)}


if __name__ == "__main__":
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    threading.Thread(target=sampler, daemon=True).start()
    time.sleep(10)
    reqs = []
    mode = sys.argv[2] if len(sys.argv) > 2 else "long+images"
    if mode == "long+images":
        phase[0] = "4 long prompts"
        with cf.ThreadPoolExecutor(4) as ex:
            reqs += list(ex.map(lambda a: long_req(*a), [(65000, 1), (100000, 2), (150000, 3), (200000, 4)]))
        print(json.dumps(reqs[-4:]), flush=True)
    steps = [(c, 8) for c in (1, 2, 4)] if mode == "images" else [(4, n) for n in (8, 16, 24)]
    for k, (conc, n) in enumerate(steps):
        phase[0] = f"{conc} x {n} images"
        with cf.ThreadPoolExecutor(conc) as ex:
            got = list(ex.map(lambda s: image_req(n, s), [10 * k + i for i in range(conc)]))
        reqs += got
        print(phase[0], json.dumps(got), flush=True)
        if any(g["status"] != 200 for g in got):
            break
        time.sleep(20)
    phase[0] = "after"
    time.sleep(15)
    stop.set()
    time.sleep(2)
    eo = subprocess.run(["journalctl", "-u", "earlyoom", "--since", started, "--no-pager", "-q"],
                        capture_output=True, text=True).stdout.splitlines()
    weo = subprocess.run(["ssh", "-n", "-o", "BatchMode=yes", "10.100.80.2",
                          f"journalctl -u earlyoom --since '{started}' --no-pager -q"],
                         capture_output=True, text=True).stdout.splitlines()
    phases = list(dict.fromkeys(r[3] for r in rows))
    report = {"tag": TAG, "started": started, "requests": reqs, "phases": {p: phase_min(p) for p in phases},
              "earlyoom_head": [line for line in eo if "sending" in line.lower() or "low memory" in line.lower()][:20],
              "earlyoom_worker": [line for line in weo if "sending" in line.lower() or "low memory" in line.lower()][:20]}
    (OUT / f"oom-{TAG}.csv").write_text("t,head_kib,worker_kib,phase\n" + "\n".join(",".join(map(str, r)) for r in rows))
    (OUT / f"oom-{TAG}.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report["phases"], indent=1))
    print("earlyoom head:", report["earlyoom_head"][:3], "worker:", report["earlyoom_worker"][:3])
