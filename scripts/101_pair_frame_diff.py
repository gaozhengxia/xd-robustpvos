"""Frame-by-frame greedy vs dagrs trace for one sequence (read-only).

    python scripts/101_pair_frame_diff.py <csv> <seq> [modeA] [modeB]

Prints only frames where |J&A - J&B| exceeds a threshold, plus the
aggregate, so a sudden collapse is easy to localise.
"""
import csv
import os
import sys
from collections import defaultdict

csv_path, seq = sys.argv[1], sys.argv[2]
mA = sys.argv[3] if len(sys.argv) > 3 else "greedy"
mB = sys.argv[4] if len(sys.argv) > 4 else "dagrs"

if not os.path.isabs(csv_path):
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), csv_path)

trace = defaultdict(dict)
agg = {}
with open(csv_path, newline="", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        if r["seq"] != seq:
            continue
        if r["mode"] not in (mA, mB):
            continue
        t = r["frame"]
        if t == "":
            agg[r["mode"]] = r
            continue
        trace[int(t)][r["mode"]] = r

if not trace:
    print(f"no frame rows for seq={seq}")
    sys.exit(1)

print(f"seq={seq}  arms={mA} vs {mB}  frames={len(trace)}")
for m in (mA, mB):
    a = agg.get(m)
    if a:
        print(f"  {m:<10} J={float(a['J']):.4f} F={float(a['F']):.4f} "
              f"J&F={float(a['J&F']):.4f} enc={a['n_encoder_calls']}")
print(f"{'t':>5}{'J_A':>9}{'J_B':>9}{'dJ':>9}{'opB':>5}{'vacB':>9}")
print("-" * 46)
big = 0
for t in sorted(trace):
    a, b = trace[t].get(mA), trace[t].get(mB)
    if not a or not b:
        continue
    ja, jb = float(a["J_frame"]), float(b["J_frame"])
    d = jb - ja
    if abs(d) > 0.02:
        big += 1
        print(f"{t:>5}{ja:>9.4f}{jb:>9.4f}{d:>+9.4f}"
              f"{b['selected_op']:>5}{b['vacuity']:>9}")
print("-" * 46)
print(f"frames with |dJ|>0.02 : {big}/{len(trace)}")
jA = [float(trace[t][mA]["J_frame"]) for t in sorted(trace) if mA in trace[t]]
jB = [float(trace[t][mB]["J_frame"]) for t in sorted(trace) if mB in trace[t]]
if jA and jB:
    print(f"mean J_frame  {mA}={sum(jA)/len(jA):.4f}  {mB}={sum(jB)/len(jB):.4f}")
