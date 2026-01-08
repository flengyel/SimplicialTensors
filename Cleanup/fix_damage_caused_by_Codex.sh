#!/bin/bash
set -euo pipefail

# 1. Create experiments dir if it doesn't exist (it should)
mkdir -p experiments

# List of files that are actually scripts, currently hiding in src/
# I identified these based on your uploads and typical patterns.
SCRIPTS=(
    "count_standard_basis_tensors.py"
    "verify_injective_face_map.py"
    "verify_degenerate_preference.py"
    "verify_degenerate_preference_range_tensor.py"
    "degenerate_counterexample.py"
    "discrepancy_test.py"
    "fixed_index_face_independence.py"
    "homotopy_constraint_verification.py"
    "homotopy_constraint_verification_with_independence_test.py"
    "n_cycle_conjugation.py"
    "n_hypergroupoid_conjecture.py"
    "petersen_permutation_test.py"
    "random_zero_one_mask_cocycle_test.py"
)

echo "--- Moving Logic from src/ to experiments/ ---"

for file in "${SCRIPTS[@]}"; do
    src_path="src/simplicial_tensors/$file"
    dest_path="experiments/$file"
    trampoline_path="examples/$file"

    if [[ -f "$src_path" ]]; then
        echo "Moving $src_path -> $dest_path"
        mv "$src_path" "$dest_path"

        # 2. Fix the imports
        # Scripts inside the package used relative imports (from .tensor_ops).
        # Scripts outside must use absolute imports (from simplicial_tensors.tensor_ops).
        # We also strip the logging setup if it duplicates standard behavior, but for now we just fix imports.
        sed -i 's/from \.tensor_ops/from simplicial_tensors.tensor_ops/g' "$dest_path"
        sed -i 's/from \./from simplicial_tensors./g' "$dest_path"
    else
        echo "Warning: $src_path not found, skipping."
    fi

    # 3. Delete the trampoline in examples/ if it exists
    if [[ -f "$trampoline_path" ]]; then
        echo "Deleting trampoline $trampoline_path"
        rm "$trampoline_path"
    fi
done

# 4. Cleanup the "Hallucination" file if it exists
if [[ -f "src/simplicial_tensors/Gemini_hallicination.py" ]]; then
    echo "Deleting src/simplicial_tensors/Gemini_hallicination.py"
    rm "src/simplicial_tensors/Gemini_hallicination.py"
fi

echo "--- Cleanup Complete ---"
echo "Your src/ directory is now lighter, and experiments/ contains the runnable code."