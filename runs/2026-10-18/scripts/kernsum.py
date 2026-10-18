"""kernsum.py CSV [N]: top kernels of an nsys cuda_gpu_kern_sum CSV, with share of total GPU kernel time."""
import csv
import sys

rows = list(csv.DictReader(open(sys.argv[1])))
n = int(sys.argv[2]) if len(sys.argv) > 2 else 15
tot = sum(float(r["Total Time (ns)"]) for r in rows)
print(f"total GPU kernel time {tot / 1e9:.2f} s, {len(rows)} kernels")
for r in rows[:n]:
    print("%5.1f%%  %6.2fs  n=%6s  avg=%8.1fus  %s" % (
        float(r["Time (%)"]), float(r["Total Time (ns)"]) / 1e9, r["Instances"],
        float(r["Avg (ns)"]) / 1e3, r["Name"][:100]))
