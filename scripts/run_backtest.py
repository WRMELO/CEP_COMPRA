#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mechanisms", nargs="+", required=True)
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--out", required=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    mechs = {m.upper() for m in args.mechanisms}
    if {"M0", "M1", "M3", "M4", "M5", "M6"}.issubset(mechs):
        repo_root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(repo_root))
        from tools.task_021_m6_runner import run_task021
        run_task021(out_dir=args.out, start=args.start, end=args.end)
        return 0
    if {"M0", "M1", "M3", "M4", "M5"}.issubset(mechs):
        repo_root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(repo_root))
        from tools.task_020_m5_runner import run_task020
        run_task020(out_dir=args.out, start=args.start, end=args.end)
        return 0
    if {"M0", "M1", "M3", "M4"}.issubset(mechs):
        repo_root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(repo_root))
        from tools.task_019_m4_runner import run_task019
        run_task019(out_dir=args.out, start=args.start, end=args.end)
        return 0

    print("Use --mechanisms M0 M1 M3 M4 (task019), M0 M1 M3 M4 M5 (task020) ou M0 M1 M3 M4 M5 M6 (task021).")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
