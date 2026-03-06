#!/usr/bin/env python3
"""Sweep --override-mass for blue 1 and blue 2 over a mesh and collect blue win rates (FFN)."""

import csv
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

# Mesh parameters
MASS_MIN = 0.0306
MASS_MAX = 0.0506
MASS_STEP = 0.002

# Run configuration — only need to update these two
EXPERIMENT = "action_penalty"
CHECKPOINT = "run_20260223130705"

# Output CSV auto-derived from run folder
OUTPUT_CSV = f"results/ffn/{EXPERIMENT}/results/{CHECKPOINT}/mass_sweep_ffn.csv"

# Eval command template
BASE_CMD = [
    sys.executable, "scripts/eval_mappo_ffn.py",
    "--experiment", EXPERIMENT,
    "--checkpoint", CHECKPOINT,
    "--n-episodes", "1000",
    "--level", "9",
    "--n-worlds", "100",
    "--deterministic",
    "--no-domain-rand",
    "--no-disturbance",
    "--step", "4000k",
]


def main():
    masses = np.round(np.arange(MASS_MIN, MASS_MAX + MASS_STEP / 2, MASS_STEP), 4)
    total = len(masses) ** 2
    print(f"Mass sweep: {len(masses)} values from {masses[0]} to {masses[-1]}")
    print(f"Total runs: {total}")

    out_path = Path(OUTPUT_CSV)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Check if partial results exist to allow resuming
    completed = set()
    if out_path.exists():
        with open(out_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                completed.add((float(row["blue1_mass"]), float(row["blue2_mass"])))
        print(f"Resuming: {len(completed)}/{total} already completed")

    write_header = not out_path.exists() or len(completed) == 0

    with open(out_path, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if write_header:
            writer.writerow(["blue1_mass", "blue2_mass", "blue_win_rate", "red_win_rate", "timeout_rate", "n_episodes"])

        run_idx = 0
        for m1 in masses:
            for m2 in masses:
                run_idx += 1
                if (m1, m2) in completed:
                    continue

                with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
                    tmp_json = tmp.name

                cmd = BASE_CMD + [
                    "--override-mass", f"{m1}", f"{m2}",
                    "--output", tmp_json,
                ]

                print(f"\n[{run_idx}/{total}] blue1={m1:.4f} blue2={m2:.4f}")
                t0 = time.time()
                result = subprocess.run(cmd, capture_output=True, text=True)

                if result.returncode != 0:
                    print(f"  FAILED (exit {result.returncode})")
                    print(result.stderr[-500:] if result.stderr else "no stderr")
                    continue

                elapsed = time.time() - t0
                try:
                    with open(tmp_json, "r") as f:
                        metrics = json.load(f)
                    blue_wr = metrics["termination_reasons"]["blue_won"]
                    red_wr = metrics["termination_reasons"]["red_won"]
                    timeout_r = metrics["termination_reasons"]["timeout"]
                    n_ep = metrics["n_episodes"]
                    writer.writerow([f"{m1:.4f}", f"{m2:.4f}", f"{blue_wr:.4f}", f"{red_wr:.4f}", f"{timeout_r:.4f}", n_ep])
                    csvfile.flush()
                    print(f"  blue_win={blue_wr*100:.1f}% red_win={red_wr*100:.1f}% timeout={timeout_r*100:.1f}% ({elapsed:.1f}s)")
                except Exception as e:
                    print(f"  ERROR parsing results: {e}")
                finally:
                    Path(tmp_json).unlink(missing_ok=True)

    print(f"\nSweep complete. Results saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
