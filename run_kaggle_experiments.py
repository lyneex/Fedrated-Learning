"""Run both separate Kaggle cardiovascular experiments with one command.

Runs clinical-tabular baseline + FedAvg first, then ECG-image baseline + FedAvg.
The input mode is passed through the environment, so source configuration is
not edited between runs and results remain isolated under ``runs/<mode>/``.
"""
import os
import subprocess
import sys


def run(mode: str, script: str) -> None:
    environment = os.environ.copy()
    environment["DATA_MODE"] = mode
    print(f"\n{'=' * 70}\nStarting {mode}: {script}\n{'=' * 70}")
    subprocess.run([sys.executable, script], check=True, env=environment)


if __name__ == "__main__":
    for mode in ("cardiovascular_tabular", "ecg_image"):
        run(mode, "baseline.py")
        run(mode, "federated_main.py")
    print("\nAll Kaggle experiments completed.")
