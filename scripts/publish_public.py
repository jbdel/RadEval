#!/usr/bin/env python3
"""Publish the public version of RadEval to jbdel/RadEval.

This script:
  1. Clones the PUBLIC repo (jbdel/RadEval)
  2. Replaces its working tree with the current private repo state
  3. Strips private metric packages, tests, and registry entries
  4. Verifies no private symbols leak, then commits and pushes

Private files never enter any commit in the public repo's history.

Private metrics are listed in PRIVATE_METRICS. Each name corresponds to:
  - RadEval/metrics/<name>/         (whole directory deleted)
  - tests/test_<name>.py            (deleted)
  - an entry inside the
    `# --- PRIVATE METRICS ---` / `# --- END PRIVATE METRICS ---`
    marker block in RadEval/metrics/_registry.py (stripped).

The script fails fast if the marker block is missing (refusing to publish
anything without registry stripping), and fails fast if any private symbol
name survives in the stripped tree.

Usage:
    python scripts/publish_public.py -m "1.0.0: description"       # dry-run
    python scripts/publish_public.py --push -m "1.0.0: description" # push
    python scripts/publish_public.py --push -m "msg" --branch main
"""
import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PUBLIC_REMOTE = "https://github.com/jbdel/RadEval.git"

PRIVATE_METRICS = [
    "f1hopprchexbert",
    "f1hopprchexbert_ct",
    "f1hopprchexbert_msk",
    "f1hopprchexbert_abd",
    "hoppr_crimson_ct",
    "nodule_eval",
]

PRIVATE_DIRS = ["scripts", ".cursor", "docs/changelog"]

PRIVATE_FILES = [
    "findings_generation_examples.csv",
    "pred_ref_epoch37_seed476104_val.jsonl",
    "run_main_ct.py",
    "run_main_cxr.py",
    "run_main.py",
    "cmd",
]

# Strings that must not appear anywhere in the stripped tree.
LEAK_PATTERNS = [
    "f1hopprchexbert",
    "hoppr_crimson_ct",
    "HopprF1CheXbert",
    "HopprCrimsonCT",
    "CRIMSON_CT",
    "nodule_eval",
    "NoduleEval",
]

# Files on the PUBLIC repo that legitimately mention private metric class names
# in docstrings/comments (e.g. the shared CheXbert base class predates the
# split; CRIMSON prose documents that CRIMSON_CT inherits its JSON helpers).
# These references are already on jbdel/RadEval from upstream and are not
# leaks — they describe sibling relationships, not importable code. Scoped to
# EXACT paths so any *new* accidental mention elsewhere still aborts.
LEAK_SCAN_ALLOWLIST = {
    "./RadEval/metrics/_chexbert_base.py",
    "./RadEval/metrics/crimson/crimson.py",
}

REGISTRY_PATH = "RadEval/metrics/_registry.py"
REGISTRY_MARKER_START = "# --- PRIVATE METRICS"
REGISTRY_MARKER_END = "# --- END PRIVATE METRICS"


def run(cmd, cwd=None, check=True):
    print(f"  $ {cmd}")
    return subprocess.run(cmd, shell=True, cwd=cwd, check=check,
                          capture_output=True, text=True)


def strip_private_metric_files(repo_dir: Path, name: str):
    metric_dir = repo_dir / "RadEval" / "metrics" / name
    if metric_dir.exists():
        shutil.rmtree(metric_dir)
        print(f"  Removed RadEval/metrics/{name}/")

    test_file = repo_dir / "tests" / f"test_{name}.py"
    if test_file.exists():
        test_file.unlink()
        print(f"  Removed tests/test_{name}.py")


def strip_registry_marker_block(repo_dir: Path):
    """Remove the private-metrics block from _registry.py.

    Fails fast if either marker is missing or if the file is unchanged
    after the edit — both conditions indicate a possible leak.
    """
    registry = repo_dir / REGISTRY_PATH
    original = registry.read_text()

    if REGISTRY_MARKER_START not in original:
        raise RuntimeError(
            f"{REGISTRY_PATH} missing start marker "
            f"'{REGISTRY_MARKER_START}'; refusing to publish — private "
            "metrics may leak into the public registry.")
    if REGISTRY_MARKER_END not in original:
        raise RuntimeError(
            f"{REGISTRY_PATH} missing end marker "
            f"'{REGISTRY_MARKER_END}'; refusing to publish — private "
            "metrics may leak into the public registry.")

    lines = original.splitlines(keepends=True)
    out = []
    inside = False
    for line in lines:
        if REGISTRY_MARKER_START in line:
            inside = True
            continue
        if REGISTRY_MARKER_END in line:
            inside = False
            continue
        if not inside:
            out.append(line)
    stripped = "".join(out)

    if stripped == original:
        raise RuntimeError(
            f"{REGISTRY_PATH} marker block stripping made no changes; "
            "refusing to publish — private metrics may leak.")

    registry.write_text(stripped)
    print(f"  Stripped marker block from {REGISTRY_PATH}")


def scan_for_leaks(repo_dir: Path):
    """Grep the stripped tree for private symbol names.

    Scoped strictly to repo_dir (not the active worktree) and excludes
    .git/ to avoid matching pre-merge commit history. Known-benign
    prose mentions in public files are allowed via LEAK_SCAN_ALLOWLIST.
    Any hit outside the allowlist aborts.
    """
    print("\n=== Scanning stripped tree for private symbol leaks ===")
    patterns = "|".join(LEAK_PATTERNS)
    cmd = (
        f"grep -rnIE --exclude-dir=.git --exclude-dir=__pycache__ "
        f"'{patterns}' ."
    )
    result = subprocess.run(
        cmd, shell=True, cwd=repo_dir,
        capture_output=True, text=True, check=False,
    )
    hits = [line for line in result.stdout.splitlines() if line]
    offending = []
    for hit in hits:
        # grep line format: ./path/to/file:NN:matched-content
        path = hit.split(":", 1)[0]
        if path in LEAK_SCAN_ALLOWLIST:
            continue
        offending.append(hit)

    if offending:
        print("\n".join(offending))
        raise RuntimeError(
            "Private symbol(s) leaked into stripped tree (outside the "
            "allowlist); aborting before commit/push. Investigate the "
            "matches above.")

    if hits:
        print(f"  Leak scan: {len(hits) - len(offending)} allowlisted "
              "prose reference(s) ignored; 0 unexpected hits.")
    else:
        print("  No private symbol leaks detected.")


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

        print("=== Cloning public repo ===")
        run(f"git clone {PUBLIC_REMOTE} {dest}")

        print("\n=== Clearing public working tree ===")
        for item in dest.iterdir():
            if item.name == ".git":
                continue
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()

        print("\n=== Copying private repo files ===")
        # Use `git ls-files` so only tracked files flow to public.
        # Untracked/gitignored files (plan/, tmp_pr_msg.md, data jsonls,
        # etc.) are never considered — no chance of accidental leak from
        # a working-tree file that happens to sit next to the repo.
        tracked = subprocess.run(
            "git ls-files", shell=True, cwd=src,
            capture_output=True, text=True, check=True,
        ).stdout.splitlines()
        for rel in tracked:
            src_path = src / rel
            if not src_path.is_file():
                continue
            dst_path = dest / rel
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_path, dst_path)

        print("\n=== Stripping private metric packages and tests ===")
        for name in PRIVATE_METRICS:
            strip_private_metric_files(dest, name)

        print("\n=== Stripping private registry entries ===")
        strip_registry_marker_block(dest)

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

        scan_for_leaks(dest)

        print("\n=== Committing ===")
        run("git add -A", cwd=dest)
        result = run("git diff --cached --quiet", cwd=dest, check=False)
        if result.returncode != 0:
            run(f'git commit -m "{args.message}"', cwd=dest)
            print(f"\nCommit created: {args.message}")
        else:
            print("\nNo changes to commit (public repo already up to date).")

        if args.push:
            run(f"git push origin HEAD:{args.branch}", cwd=dest)
            print(f"\nPushed to {PUBLIC_REMOTE} branch {args.branch}")
        else:
            print(f"\n[DRY RUN] Would push to {PUBLIC_REMOTE} branch {args.branch}")
            print("Run with --push to actually push.")

        print("\n--- Public repo metrics listing ---")
        run("find RadEval/metrics -type d | sort", cwd=dest)


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        sys.exit(1)
