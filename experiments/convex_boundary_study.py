#!/usr/bin/env python3
"""Controlled convex test of diagonal-boundary Tikhonov regularization.

This is a deliberately small Stage-1 experiment.  It asks a narrow question:
does the eigenspace picked out by the diagonal simplicial tensor boundary help
linear regression when the truth is, or is not, aligned with that eigenspace?

The four penalties are normalized to have mean positive eigenvalue one and use
the same validation grid:

``dstm_boundary``
    ``||bdry(W)||_F^2``, implemented by the repository's adjoint boundary.
``ridge``
    ``||W||_F^2``.
``isospectral_random``
    One fixed Haar-orthogonal conjugate of the DSTM penalty.  It has the same
    spectrum and kernel dimension, but random eigenspaces.
``grid_laplacian``
    The standard four-neighbour two-dimensional grid smoothness penalty.

Every method sees identical train/validation/test data within a replicate.
Lambda is selected using validation MSE; validation data are not used for
fitting, and the selected training-set model is evaluated once on held-out
data.  Apart from the repository's boundary implementation, the numerical
experiment uses only NumPy and the Python standard library.

Example
-------
PYTHONPATH=src python experiments/convex_boundary_study.py \
    --seeds 12 --output-prefix /tmp/convex_boundary_study
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from simplicial_tensors.adjoint_ops import lower_hodge_laplacian


CONDITIONS = ("aligned_in_kernel", "isotropic", "permuted_misaligned")
METHODS = ("dstm_boundary", "ridge", "isospectral_random", "grid_laplacian")


def dstm_penalty(matrix_size: int) -> np.ndarray:
    """Return the matrix of ``bdry_adjoint @ bdry`` on square matrices."""

    dimension = matrix_size * matrix_size
    operator = np.empty((dimension, dimension), dtype=float)
    for column in range(dimension):
        basis = np.zeros((matrix_size, matrix_size), dtype=float)
        basis.flat[column] = 1.0
        operator[:, column] = lower_hodge_laplacian(basis).ravel()
    return 0.5 * (operator + operator.T)


def grid_penalty(matrix_size: int) -> np.ndarray:
    """Return the combinatorial Laplacian of an ``n`` by ``n`` grid."""

    path = np.zeros((matrix_size, matrix_size), dtype=float)
    for index in range(matrix_size - 1):
        path[index, index] += 1.0
        path[index + 1, index + 1] += 1.0
        path[index, index + 1] -= 1.0
        path[index + 1, index] -= 1.0
    identity = np.eye(matrix_size)
    return np.kron(path, identity) + np.kron(identity, path)


def normalize_penalty(penalty: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Normalize a PSD penalty by its mean positive eigenvalue."""

    penalty = 0.5 * (penalty + penalty.T)
    eigenvalues = np.linalg.eigvalsh(penalty)
    spectral_radius = float(max(1.0, np.max(np.abs(eigenvalues))))
    tolerance = 1e-10 * spectral_radius
    if float(eigenvalues[0]) < -tolerance:
        raise ValueError(f"penalty is not positive semidefinite: {eigenvalues[0]:.3e}")
    positive = eigenvalues[eigenvalues > tolerance]
    if positive.size == 0:
        raise ValueError("penalty has no positive eigenvalues")
    scale = float(np.mean(positive))
    normalized = penalty / scale
    normalized_eigenvalues = np.linalg.eigvalsh(normalized)
    rank = int(np.count_nonzero(normalized_eigenvalues > tolerance / scale))
    diagnostics = {
        "dimension": int(penalty.shape[0]),
        "rank": rank,
        "kernel_dimension": int(penalty.shape[0] - rank),
        "normalization_scale": scale,
        "smallest_eigenvalue": float(normalized_eigenvalues[0]),
        "largest_eigenvalue": float(normalized_eigenvalues[-1]),
        "mean_positive_eigenvalue": float(
            np.mean(normalized_eigenvalues[normalized_eigenvalues > tolerance / scale])
        ),
    }
    return normalized, diagnostics


def build_penalties(
    matrix_size: int, control_seed: int
) -> tuple[dict[str, np.ndarray], dict[str, Any], np.ndarray, np.ndarray]:
    """Construct normalized penalties, the DSTM kernel, and a fixed permutation."""

    raw_dstm = dstm_penalty(matrix_size)
    normalized_dstm, dstm_diagnostics = normalize_penalty(raw_dstm)
    dimension = normalized_dstm.shape[0]

    control_rng = np.random.default_rng(control_seed)
    gaussian = control_rng.normal(size=(dimension, dimension))
    orthogonal, triangular = np.linalg.qr(gaussian)
    signs = np.where(np.diag(triangular) < 0.0, -1.0, 1.0)
    orthogonal = orthogonal * signs
    random_conjugate = orthogonal @ normalized_dstm @ orthogonal.T

    normalized_grid, grid_diagnostics = normalize_penalty(
        grid_penalty(matrix_size)
    )
    ridge = np.eye(dimension)
    _, ridge_diagnostics = normalize_penalty(ridge)
    random_conjugate = 0.5 * (random_conjugate + random_conjugate.T)
    _, random_diagnostics = normalize_penalty(random_conjugate)
    # Before normalization the random control is a conjugate of the raw DSTM
    # penalty, so it uses exactly the same scaling constant.
    random_diagnostics["normalization_scale"] = dstm_diagnostics["normalization_scale"]

    penalties: dict[str, np.ndarray] = {
        "dstm_boundary": normalized_dstm,
        "ridge": ridge,
        "isospectral_random": random_conjugate,
        "grid_laplacian": normalized_grid,
    }

    diagnostics: dict[str, Any] = {
        "dstm_boundary": dstm_diagnostics,
        "ridge": ridge_diagnostics,
        "isospectral_random": random_diagnostics,
        "grid_laplacian": grid_diagnostics,
    }

    dstm_spectrum = np.linalg.eigvalsh(penalties["dstm_boundary"])
    random_spectrum = np.linalg.eigvalsh(penalties["isospectral_random"])
    diagnostics["isospectral_max_eigenvalue_error"] = float(
        np.max(np.abs(dstm_spectrum - random_spectrum))
    )

    tolerance = 1e-9 * max(1.0, float(dstm_spectrum[-1]))
    _, eigenvectors = np.linalg.eigh(penalties["dstm_boundary"])
    kernel = eigenvectors[:, dstm_spectrum <= tolerance]
    if kernel.shape[1] == 0:
        raise RuntimeError("the DSTM penalty unexpectedly has a trivial kernel")

    permutation = control_rng.permutation(dimension)
    return penalties, diagnostics, kernel, permutation


def normalize_vector(vector: np.ndarray, target_norm: float) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-14:
        raise RuntimeError("cannot normalize a numerically zero truth vector")
    return vector * (target_norm / norm)


def make_truths(
    rng: np.random.Generator,
    kernel: np.ndarray,
    permutation: np.ndarray,
    signal_norm: float,
) -> dict[str, np.ndarray]:
    """Return paired aligned, generic, and coordinate-permuted truths."""

    aligned_coordinates = rng.normal(size=kernel.shape[1])
    aligned = normalize_vector(kernel @ aligned_coordinates, signal_norm)
    isotropic = normalize_vector(rng.normal(size=kernel.shape[0]), signal_norm)
    permuted = aligned[permutation]
    return {
        "aligned_in_kernel": aligned,
        "isotropic": isotropic,
        "permuted_misaligned": permuted,
    }


def fit_penalized(
    design: np.ndarray,
    response: np.ndarray,
    penalty: np.ndarray,
    regularization: float,
) -> np.ndarray:
    """Solve mean squared loss plus ``regularization * beta.T P beta``."""

    sample_count = design.shape[0]
    gram = design.T @ design / sample_count
    rhs = design.T @ response / sample_count
    system = gram + regularization * penalty
    if regularization == 0.0:
        return np.linalg.lstsq(design, response, rcond=None)[0]
    # A semidefinite penalty can leave the objective non-strictly convex when
    # the training set is small.  ``solve`` is unsafe here: a numerically
    # singular system may not raise and can return a huge arbitrary minimizer.
    # The Moore--Penrose solution makes the tie rule explicit and reproducible.
    return np.linalg.lstsq(system, rhs, rcond=None)[0]


def mean_squared_error(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.square(prediction - target)))


def select_regularization(
    train_design: np.ndarray,
    train_response: np.ndarray,
    validation_design: np.ndarray,
    validation_response: np.ndarray,
    penalty: np.ndarray,
    lambda_grid: Iterable[float],
) -> tuple[float, float]:
    """Select lambda solely by validation MSE, breaking ties toward smaller lambda."""

    candidates: list[tuple[float, float]] = []
    for regularization in lambda_grid:
        estimate = fit_penalized(
            train_design, train_response, penalty, float(regularization)
        )
        validation_mse = mean_squared_error(
            validation_design @ estimate, validation_response
        )
        candidates.append((validation_mse, float(regularization)))
    validation_mse, regularization = min(candidates, key=lambda item: (item[0], item[1]))
    return regularization, validation_mse


def run_study(args: argparse.Namespace) -> dict[str, Any]:
    """Run all paired replicates and return a JSON-serializable result."""

    penalties, penalty_diagnostics, kernel, permutation = build_penalties(
        args.matrix_size, args.control_seed
    )
    dimension = args.matrix_size**2
    lambda_grid = np.concatenate(
        ([0.0], np.logspace(args.lambda_min_exp, args.lambda_max_exp, args.lambda_count))
    )
    rows: list[dict[str, Any]] = []

    for replicate in range(args.seeds):
        seed = args.seed_start + replicate
        sequence = np.random.SeedSequence([args.base_seed, seed])
        design_seed, truth_seed, noise_seed = sequence.spawn(3)
        design_rng = np.random.default_rng(design_seed)
        truth_rng = np.random.default_rng(truth_seed)
        noise_rng = np.random.default_rng(noise_seed)

        total_samples = args.n_train + args.n_val + args.n_test
        design = design_rng.normal(size=(total_samples, dimension))
        noise = noise_rng.normal(scale=args.noise_std, size=total_samples)
        train_end = args.n_train
        validation_end = args.n_train + args.n_val
        train_design = design[:train_end]
        validation_design = design[train_end:validation_end]
        test_design = design[validation_end:]
        truths = make_truths(truth_rng, kernel, permutation, args.signal_norm)
        for condition in CONDITIONS:
            truth = truths[condition]
            response = design @ truth + noise
            train_response = response[:train_end]
            validation_response = response[train_end:validation_end]
            test_response = response[validation_end:]
            for method in METHODS:
                penalty = penalties[method]
                regularization, validation_mse = select_regularization(
                    train_design,
                    train_response,
                    validation_design,
                    validation_response,
                    penalty,
                    lambda_grid,
                )
                estimate = fit_penalized(
                    train_design,
                    train_response,
                    penalty,
                    regularization,
                )
                rows.append(
                    {
                        "seed": seed,
                        "condition": condition,
                        "method": method,
                        "selected_lambda": regularization,
                        "validation_mse": validation_mse,
                        "test_mse": mean_squared_error(
                            test_design @ estimate, test_response
                        ),
                        "coefficient_mse": mean_squared_error(estimate, truth),
                        "truth_penalty_energy": float(truth @ penalty @ truth),
                        "estimate_penalty_energy": float(estimate @ penalty @ estimate),
                    }
                )

    summary = summarize(rows)
    return {
        "config": {
            "matrix_size": args.matrix_size,
            "dimension": dimension,
            "n_train": args.n_train,
            "n_val": args.n_val,
            "n_test": args.n_test,
            "noise_std": args.noise_std,
            "signal_norm": args.signal_norm,
            "expected_signal_variance": args.signal_norm**2,
            "noise_variance": args.noise_std**2,
            "signal_to_noise_variance_ratio": (
                args.signal_norm**2 / args.noise_std**2
                if args.noise_std > 0.0
                else None
            ),
            "seeds": args.seeds,
            "seed_start": args.seed_start,
            "base_seed": args.base_seed,
            "control_seed": args.control_seed,
            "lambda_grid": [float(value) for value in lambda_grid],
            "penalty_normalization": "mean positive eigenvalue equals one",
            "selection_rule": "minimum validation MSE; ties use smaller lambda",
            "fit_protocol": (
                "fit on training data; validation selects lambda only; "
                "held-out test data used once"
            ),
        },
        "penalty_diagnostics": penalty_diagnostics,
        "summary": summary,
        "runs": rows,
    }


def standard_error(values: np.ndarray) -> float:
    if values.size <= 1:
        return 0.0
    return float(np.std(values, ddof=1) / np.sqrt(values.size))


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate outcomes and paired test-MSE differences versus ridge."""

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    ridge_by_pair: dict[tuple[str, int], float] = {}
    for row in rows:
        grouped[(row["condition"], row["method"])].append(row)
        if row["method"] == "ridge":
            ridge_by_pair[(row["condition"], row["seed"])] = row["test_mse"]

    summaries: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        for method in METHODS:
            group = grouped[(condition, method)]
            test_values = np.asarray([row["test_mse"] for row in group], dtype=float)
            coefficient_values = np.asarray(
                [row["coefficient_mse"] for row in group], dtype=float
            )
            truth_energy_values = np.asarray(
                [row["truth_penalty_energy"] for row in group], dtype=float
            )
            lambdas = np.asarray([row["selected_lambda"] for row in group], dtype=float)
            paired_differences = np.asarray(
                [
                    row["test_mse"]
                    - ridge_by_pair[(row["condition"], row["seed"])]
                    for row in group
                ],
                dtype=float,
            )
            difference_mean = float(np.mean(paired_differences))
            difference_se = standard_error(paired_differences)
            summaries.append(
                {
                    "condition": condition,
                    "method": method,
                    "replicates": int(test_values.size),
                    "test_mse_mean": float(np.mean(test_values)),
                    "test_mse_std": float(
                        np.std(test_values, ddof=1) if test_values.size > 1 else 0.0
                    ),
                    "test_mse_se": standard_error(test_values),
                    "coefficient_mse_mean": float(np.mean(coefficient_values)),
                    "truth_penalty_energy_mean": float(np.mean(truth_energy_values)),
                    "selected_lambda_median": float(np.median(lambdas)),
                    "paired_test_mse_difference_vs_ridge_mean": difference_mean,
                    "paired_test_mse_difference_vs_ridge_se": difference_se,
                    "paired_difference_approx_95pct_interval": [
                        difference_mean - 1.96 * difference_se,
                        difference_mean + 1.96 * difference_se,
                    ],
                }
            )
    return summaries


def write_outputs(result: dict[str, Any], output_prefix: Path) -> tuple[Path, Path]:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = Path(f"{output_prefix}.json")
    csv_path = Path(f"{output_prefix}.csv")
    json_result = {key: value for key, value in result.items() if key != "runs"}
    json_result["runs_csv"] = csv_path.name
    json_path.write_text(
        json.dumps(json_result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    rows = result["runs"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return json_path, csv_path


def print_summary(summary: list[dict[str, Any]]) -> None:
    print(
        "condition               method                 "
        "test MSE (mean +/- SE)   delta vs ridge"
    )
    for row in summary:
        print(
            f"{row['condition']:<23} "
            f"{row['method']:<22} "
            f"{row['test_mse_mean']:.5f} +/- {row['test_mse_se']:.5f}   "
            f"{row['paired_test_mse_difference_vs_ridge_mean']:+.5f}"
        )


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-size", type=positive_int, default=8)
    parser.add_argument("--n-train", type=positive_int, default=48)
    parser.add_argument("--n-val", type=positive_int, default=128)
    parser.add_argument("--n-test", type=positive_int, default=1024)
    parser.add_argument("--noise-std", type=float, default=0.5)
    parser.add_argument("--signal-norm", type=float, default=2.0)
    parser.add_argument("--seeds", type=positive_int, default=12)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--base-seed", type=int, default=20260813)
    parser.add_argument("--control-seed", type=int, default=271828)
    parser.add_argument("--lambda-min-exp", type=float, default=-5.0)
    parser.add_argument("--lambda-max-exp", type=float, default=3.0)
    parser.add_argument("--lambda-count", type=positive_int, default=17)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("convex_boundary_study_results"),
        help="write PREFIX.json and PREFIX.csv",
    )
    args = parser.parse_args()
    if args.matrix_size < 2:
        parser.error("--matrix-size must be at least 2")
    if args.noise_std < 0.0:
        parser.error("--noise-std must be non-negative")
    if args.signal_norm <= 0.0:
        parser.error("--signal-norm must be positive")
    if args.lambda_min_exp >= args.lambda_max_exp:
        parser.error("--lambda-min-exp must be smaller than --lambda-max-exp")
    if min(args.seed_start, args.base_seed, args.control_seed) < 0:
        parser.error("seed arguments must be non-negative")
    return args


def main() -> None:
    args = parse_args()
    result = run_study(args)
    print_summary(result["summary"])
    json_path, csv_path = write_outputs(result, args.output_prefix)
    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
