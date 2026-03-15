#!/usr/bin/env python3
"""Publish the public version of RadEval to jbdel/RadEval.

This script:
  1. Clones the PUBLIC repo (jbdel/RadEval)
  2. Replaces its working tree with the current private repo state
  3. Strips private metrics (files, imports, tests, references)
  4. Commits with a real message and pushes normally

Private files never enter any commit in the public repo's history.

Private metrics are defined in PRIVATE_METRICS below. Each entry specifies
the metric folder to delete, the flag name in RadEval.__init__, the test
file, and any other files that reference it.

Usage:
    python scripts/publish_public.py -m "v0.1.5: description"       # dry-run
    python scripts/publish_public.py --push -m "v0.1.5: description" # push
    python scripts/publish_public.py --push -m "msg" --branch main
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
        "flag": "do_f1hopprchexbert",
        "display_name": "F1HopprCheXbert",
    },
    {
        "name": "f1hopprchexbert_ct",
        "metric_dir": "RadEval/metrics/f1hopprchexbert_ct",
        "test_file": "tests/test_f1hopprchexbert_ct.py",
        "flag": "do_f1hopprchexbert_ct",
        "display_name": "F1HopprCheXbertCT",
    },
]

PRIVATE_DIRS = ["scripts", ".cursor"]

PRIVATE_FILES = [
    "findings_generation_examples.csv",
    "pred_ref_epoch37_seed476104_val.jsonl",
    "run_main_ct.py",
    "run_main_cxr.py",
    "run_main.py",
    "cmd",
]


def run(cmd, cwd=None, check=True):
    print(f"  $ {cmd}")
    return subprocess.run(cmd, shell=True, cwd=cwd, check=check,
                          capture_output=True, text=True)


def strip_flag_from_init(radeval_py: Path, flag: str, display_name: str):
    """Remove all references to a private metric flag from RadEval.py."""
    text = radeval_py.read_text()

    # Remove constructor parameter line:  do_<flag>=False,
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
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--push", action="store_true",
                        help="Actually push to the public remote (default: dry-run)")
    parser.add_argument("--branch", default="main",
                        help="Branch to push to (default: main)")
    parser.add_argument("-m", "--message", required=True,
                        help="Commit message for the public repo")
    args = parser.parse_args()

    src = Path(__file__).resolve().parent.parent
    print(f"Source (private) repo: {src}")

    with tempfile.TemporaryDirectory(prefix="radeval_public_") as tmp:
        dest = Path(tmp) / "RadEval"
        print(f"Working copy: {dest}\n")

        # 1. Clone the PUBLIC repo (keeps its own clean history)
        print("=== Cloning public repo ===")
        run(f"git clone {PUBLIC_REMOTE} {dest}")

        # 2. Clear working tree (keep .git/)
        print("\n=== Clearing public working tree ===")
        for item in dest.iterdir():
            if item.name == ".git":
                continue
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()

        # 3. Copy private repo state (minus .git and untracked data files)
        print("\n=== Copying private repo files ===")
        ignore = shutil.ignore_patterns(
            ".git", "__pycache__", "*.pyc", ".pytest_cache",
            "*.egg-info", "dist", "build",
        )
        shutil.copytree(src, dest, dirs_exist_ok=True, ignore=ignore)

        # 4. Strip private metrics
        print("\n=== Stripping private metrics ===")
        for metric in PRIVATE_METRICS:
            strip_private_metric(dest, metric)

        # 5. Remove private directories and files
        print("\n=== Removing private directories/files ===")
        for private_dir in PRIVATE_DIRS:
            d = dest / private_dir
            if d.exists():
                shutil.rmtree(d)
                print(f"  Removed {private_dir}/")

        for private_file in PRIVATE_FILES:
            f = dest / private_file
            if f.exists():
                f.unlink()
                print(f"  Removed {private_file}")

        # 6. Stage and commit
        print("\n=== Committing ===")
        run("git add -A", cwd=dest)
        result = run("git diff --cached --quiet", cwd=dest, check=False)
        if result.returncode != 0:
            run(f'git commit -m "{args.message}"', cwd=dest)
            print(f"\nCommit created: {args.message}")
        else:
            print("\nNo changes to commit (public repo already up to date).")

        # 7. Push
        if args.push:
            run(f"git push origin HEAD:{args.branch}", cwd=dest)
            print(f"\nPushed to {PUBLIC_REMOTE} branch {args.branch}")
        else:
            print(f"\n[DRY RUN] Would push to {PUBLIC_REMOTE} branch {args.branch}")
            print("Run with --push to actually push.")

        # Show what the public repo looks like
        print("\n--- Public repo metrics listing ---")
        run("find RadEval/metrics -type d | sort", cwd=dest)


if __name__ == "__main__":
    main()
