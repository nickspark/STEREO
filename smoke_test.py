from __future__ import annotations

import subprocess
import sys
import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a one-epoch CPU smoke test on a released training table.")
    parser.add_argument("--data", required=True, help="Path to the released labelled training CSV or XLSX.")
    args = parser.parse_args()
    for variant in ("stereo",):
        cmd = [
            sys.executable,
            "train.py",
            "--variant",
            variant,
            "--data",
            args.data,
            "--epochs",
            "1",
            "--batch-size",
            "4",
            "--samples",
            "32",
            "--device",
            "cpu",
            "--summary-path",
            f"outputs/{variant}_smoke_summary.json",
            "--model-save-path",
            f"outputs/{variant}_smoke_model.pt",
            "--log-dir",
            f"outputs/{variant}_smoke_run",
        ]
        code = subprocess.call(cmd)
        if code != 0:
            raise SystemExit(code)


if __name__ == "__main__":
    main()
