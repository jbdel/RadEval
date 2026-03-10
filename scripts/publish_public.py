#!/usr/bin/env python3
"""Publish the public version of RadEval to jbdel/RadEval.

This script:
  1. Clones the current repo into a temp directory
  2. Removes private metrics (files, imports, tests, references)
  3. Force-pushes the result to the public remote

Private metrics are defined in PRIVATE_METRICS below. Each entry specifies
the metric folder to delete, the flag name in RadEval.__init__, the test
file, and any other files that reference it.

Usage:
    python scripts/publish_public.py                     # dry-run (default)
    python scripts/publish_public.py --push              # actually push
    python scripts/publish_public.py --push --branch main
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PUBLIC_REMOTE = "https://github.com/jbdel/RadEval.git"

PRIVATE_METRICS = [
    {
        "name": "hoppr_f1chexbert",
        "metric_dir": "RadEval/metrics/hoppr_f1chexbert",
        "test_file": "tests/test_hopprf1chexbert.py",
        "flag": "do_hopprchexbert",
        "display_name": "HopprCheXbert",
    },
    {
        "name": "hoppr_f1chexbert_ct",
        "metric_dir": "RadEval/metrics/hoppr_f1chexbert_ct",
        "test_file": "tests/test_hoppr_f1chexbert_ct.py",
        "flag": "do_hoppr_f1chexbert_ct",
        "display_name": "HopprF1CheXbertCT",
    },
]


def run(cmd, cwd=None, check=True):
    print(f"  $ {cmd}")
    return subprocess.run(cmd, shell=True, cwd=cwd, check=check,
                          capture_output=True, text=True)


def strip_flag_from_init(radeval_py: Path, flag: str, display_name: str):
    """Remove all references to a private metric flag from RadEval.py."""
    text = radeval_py.read_text()

    # Remove constructor parameter line:  do_hopprchexbert=False,
    text = re.sub(rf'\s*{flag}=False,\n', '\n', text)

    # Remove self.do_X = do_X assignment
    text = re.sub(rf'\s*self\.{flag} = {flag}\n', '\n', text)

    # Remove the init block: if self.do_X: ... (try/except or simple)
    text = re.sub(
        rf'\n        if self\.{flag}:.*?(?=\n        if self\.|(?=\n        self\.metric_keys))',
        '\n', text, flags=re.DOTALL)

    # Remove metric_keys block
    text = re.sub(
        rf'\n        if self\.{flag}:\n.*?(?=\n        if self\.|(?=\n\n))',
        '', text, flags=re.DOTALL)

    # Remove from enabled list
    text = re.sub(rf'.*if self\.{flag}:.*enabled.*\n', '', text)

    # Remove from compute_scores
    text = re.sub(
        rf'\n            # -+\n            if self\.{flag}:.*?(?=\n            # -+|\n\n        return scores)',
        '', text, flags=re.DOTALL)

    # Remove from main() example
    text = re.sub(rf'\s*{flag}=True,\n', '\n', text)

    radeval_py.write_text(text)


def strip_private_metric(repo_dir: Path, metric: dict):
    name = metric["name"]
    print(f"\n--- Stripping private metric: {name} ---")

    metric_dir = repo_dir / metric["metric_dir"]
    if metric_dir.exists():
        shutil.rmtree(metric_dir)
        print(f"  Removed {metric['metric_dir']}/")

    test_file = repo_dir / metric["test_file"]
    if test_file.exists():
        test_file.unlink()
        print(f"  Removed {metric['test_file']}")

    radeval_py = repo_dir / "RadEval" / "RadEval.py"
    strip_flag_from_init(radeval_py, metric["flag"], metric["display_name"])
    print(f"  Stripped {metric['flag']} from RadEval.py")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--push", action="store_true",
                        help="Actually push to the public remote (default: dry-run)")
    parser.add_argument("--branch", default="main",
                        help="Branch to push to (default: main)")
    args = parser.parse_args()

    src = Path(__file__).resolve().parent.parent
    print(f"Source repo: {src}")

    with tempfile.TemporaryDirectory(prefix="radeval_public_") as tmp:
        dest = Path(tmp) / "RadEval"
        print(f"Working copy: {dest}\n")

        # Clone the current repo (local copy, preserves commits)
        run(f"git clone {src} {dest}")

        # Strip private metrics
        for metric in PRIVATE_METRICS:
            strip_private_metric(dest, metric)

        # Also remove this script itself from the public repo
        scripts_dir = dest / "scripts"
        if scripts_dir.exists():
            shutil.rmtree(scripts_dir)
            print("  Removed scripts/")

        # Commit the stripping
        run("git add -A", cwd=dest)
        result = run("git diff --cached --quiet", cwd=dest, check=False)
        if result.returncode != 0:
            run('git commit -m "Prepare public release (strip private metrics)"',
                cwd=dest)
            print("\nPublic release commit created.")
        else:
            print("\nNo changes to commit (already clean).")

        if args.push:
            run(f"git remote set-url origin {PUBLIC_REMOTE}", cwd=dest)
            run(f"git push --force origin HEAD:{args.branch}", cwd=dest)
            print(f"\nPushed to {PUBLIC_REMOTE} branch {args.branch}")
        else:
            print(f"\n[DRY RUN] Would push to {PUBLIC_REMOTE} branch {args.branch}")
            print("Run with --push to actually push.")

        # Show what the public repo looks like
        print("\n--- Public repo file listing ---")
        run("find RadEval/metrics -type d | sort", cwd=dest)


if __name__ == "__main__":
    main()
