#!/usr/bin/env bash
set -euo pipefail

# Preview package-to-experiment moves; --apply performs the listed moves.
# Existing destinations and unsupported relative imports stop the operation.
# Source files and example entry points are never deleted.
task_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
task_repo_root="$(git -C "$task_script_dir" rev-parse --show-toplevel)"

python3 - "$task_repo_root" "$@" <<'PY'
import argparse
import ast
from pathlib import Path
import re
import subprocess
import sys

parser = argparse.ArgumentParser(description="Move named experiment scripts out of the library package.")
parser.add_argument("--apply", action="store_true", help="Apply the previewed moves and tensor_ops import changes")
args = parser.parse_args(sys.argv[2:])
root = Path(sys.argv[1]).resolve()
names = (
    "count_standard_basis_tensors.py",
    "verify_injective_face_map.py",
    "verify_degenerate_preference.py",
    "verify_degenerate_preference_range_tensor.py",
    "degenerate_counterexample.py",
    "discrepancy_test.py",
    "fixed_index_face_independence.py",
    "homotopy_constraint_verification.py",
    "homotopy_constraint_verification_with_independence_test.py",
    "n_cycle_conjugation.py",
    "n_hypergroupoid_conjecture.py",
    "petersen_permutation_test.py",
    "random_zero_one_mask_cocycle_test.py",
)
plans = []
for name in names:
    source = root / "src" / "simplicial_tensors" / name
    destination = root / "experiments" / name
    if not source.exists():
        continue
    if source.is_symlink() or not source.is_file():
        raise SystemExit(f"Refusing nonregular source: {source}")
    if destination.exists() or destination.is_symlink():
        raise SystemExit(f"Destination already exists: {destination}")
    subprocess.run(["git", "-C", str(root), "ls-files", "--error-unmatch", str(source)],
                   check=True, stdout=subprocess.DEVNULL)
    text = source.read_text(encoding="utf-8")
    updated = re.sub(r"(?m)^([ \t]*)from \.tensor_ops(\s+import\b)",
                     r"\1from simplicial_tensors.tensor_ops\2", text)
    for node in ast.walk(ast.parse(updated, filename=str(source))):
        if isinstance(node, ast.ImportFrom) and node.level:
            raise SystemExit(f"Relative import needs manual review: {source}:{node.lineno}")
    plans.append((source, destination, updated))

for source, destination, _ in plans:
    print(f"{source.relative_to(root)} -> {destination.relative_to(root)}")
if not plans:
    print("No listed package scripts need moving.")
elif not args.apply:
    print("Preview only. Use --apply to perform these moves.")
else:
    for source, destination, updated in plans:
        destination.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "-C", str(root), "mv", "--", str(source), str(destination)], check=True)
        destination.write_text(updated, encoding="utf-8")
    print("Moves applied. Inspect the Git diff and run the experiment checks.")
PY
