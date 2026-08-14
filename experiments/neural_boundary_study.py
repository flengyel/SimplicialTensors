"""Controlled CPU study of tensor-boundary feedback on a small neural net.

This experiment deliberately separates three questions:

1. Does the raw diagonal DSTM operator respect function-preserving hidden
   neuron relabeling?  (It should not, and this is a falsifier for generic
   dense-weight use.)
2. Does the exact finite DSTM Sobolev filter improve optimization beyond an
   ordinary grid smoother or an exact-spectrum randomly conjugated control?
3. Does a typed architecture-incidence balance penalty behave equivariantly?

The script uses only NumPy, SciPy, and scikit-learn.  It performs an equal-size
validation search for every method, freezes the selected configurations, and
runs paired confirmation seeds.  Results are written as CSV and JSON.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.fft import dctn, idctn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from simplicial_tensors.adjoint_ops import (
    boundary_sobolev_filter,
    boundary_spectral_values,
    lower_hodge_laplacian,
)
from simplicial_tensors.architecture_ops import (
    hidden_balance_energy,
    hidden_balance_gradients,
    hidden_balance_residuals,
    path_diamond_energy,
)
from simplicial_tensors.tensor_ops import bdry


METHODS = (
    "baseline",
    "dstm_penalty",
    "matched_spectrum_penalty",
    "dstm_sobolev",
    "matched_spectrum_sobolev",
    "grid_sobolev",
    "architecture_balance",
)


@dataclass(frozen=True)
class Config:
    method: str
    learning_rate: float
    weight_decay: float = 1e-4
    strength: float = 0.0
    momentum: float = 0.9


@dataclass
class RunResult:
    method: str
    seed: int
    learning_rate: float
    weight_decay: float
    strength: float
    epochs: int
    train_nll: float
    validation_nll: float
    validation_nll_auc: float
    test_nll: float
    test_accuracy: float
    test_brier: float
    corrupted_accuracy: float
    boundary_ratio: float
    balance_ratio: float
    path_diamond_ratio: float
    wall_seconds: float


class SignedPermutation:
    """Orthogonal signed permutation used to conjugate a DSTM filter."""

    def __init__(self, shape: tuple[int, ...], rng: np.random.Generator):
        self.shape = shape
        size = int(np.prod(shape))
        self.permutation = rng.permutation(size)
        self.signs = rng.choice(np.array([-1.0, 1.0]), size=size)

    def forward(self, value: np.ndarray) -> np.ndarray:
        flat = np.asarray(value).reshape(-1)
        return (self.signs * flat[self.permutation]).reshape(self.shape)

    def adjoint(self, value: np.ndarray) -> np.ndarray:
        transformed = self.signs * np.asarray(value).reshape(-1)
        result = np.empty_like(transformed)
        result[self.permutation] = transformed
        return result.reshape(self.shape)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    return probabilities / np.sum(probabilities, axis=1, keepdims=True)


def _initialize(seed: int, input_width: int = 64, hidden_width: int = 24) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        "W1": rng.normal(size=(hidden_width, input_width)) * np.sqrt(2.0 / input_width),
        "b1": np.zeros(hidden_width),
        "W2": rng.normal(size=(10, hidden_width)) * np.sqrt(2.0 / hidden_width),
        "b2": np.zeros(10),
    }


def _forward(
    parameters: dict[str, np.ndarray], x: np.ndarray
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    preactivation = x @ parameters["W1"].T + parameters["b1"]
    hidden = np.maximum(preactivation, 0.0)
    logits = hidden @ parameters["W2"].T + parameters["b2"]
    return logits, (preactivation, hidden)


def _loss_and_gradient(
    parameters: dict[str, np.ndarray], x: np.ndarray, y: np.ndarray
) -> tuple[float, dict[str, np.ndarray]]:
    logits, (preactivation, hidden) = _forward(parameters, x)
    probabilities = _softmax(logits)
    sample_count = len(y)
    loss = -float(np.mean(np.log(probabilities[np.arange(sample_count), y] + 1e-15)))

    output_gradient = probabilities.copy()
    output_gradient[np.arange(sample_count), y] -= 1.0
    output_gradient /= sample_count
    hidden_gradient = output_gradient @ parameters["W2"]
    preactivation_gradient = hidden_gradient * (preactivation > 0.0)
    gradients = {
        "W1": preactivation_gradient.T @ x,
        "b1": np.sum(preactivation_gradient, axis=0),
        "W2": output_gradient.T @ hidden,
        "b2": np.sum(output_gradient, axis=0),
    }
    return loss, gradients


def _metrics(parameters: dict[str, np.ndarray], x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    probabilities = _softmax(_forward(parameters, x)[0])
    one_hot = np.eye(10)[y]
    return {
        "nll": -float(np.mean(np.log(probabilities[np.arange(len(y)), y] + 1e-15))),
        "accuracy": float(np.mean(np.argmax(probabilities, axis=1) == y)),
        "brier": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
    }


def _grid_sobolev_filter(value: np.ndarray, mu: float) -> np.ndarray:
    """Apply a standard separable path-graph Laplacian resolvent."""

    frequencies = dctn(value, type=2, norm="ortho")
    eigenvalues = np.zeros(value.shape)
    for axis, size in enumerate(value.shape):
        axis_values = 2.0 - 2.0 * np.cos(np.pi * np.arange(size) / size)
        reshape = [1] * value.ndim
        reshape[axis] = size
        eigenvalues = eigenvalues + axis_values.reshape(reshape)
    maximum = float(np.max(eigenvalues))
    filtered = frequencies / (1.0 + mu * eigenvalues / maximum)
    return idctn(filtered, type=2, norm="ortho")


def _apply_method(
    parameters: dict[str, np.ndarray],
    gradients: dict[str, np.ndarray],
    config: Config,
    random_operators: dict[str, SignedPermutation],
) -> dict[str, np.ndarray]:
    adjusted = {name: gradient.copy() for name, gradient in gradients.items()}

    if config.method == "dstm_penalty":
        for name in ("W1", "W2"):
            maximum = boundary_spectral_values(parameters[name].shape)[-1]
            adjusted[name] += (
                config.strength * lower_hodge_laplacian(parameters[name]) / maximum
            )
    elif config.method == "matched_spectrum_penalty":
        for name in ("W1", "W2"):
            maximum = boundary_spectral_values(parameters[name].shape)[-1]
            operator = random_operators[name]
            transformed = operator.forward(parameters[name])
            penalty_gradient = lower_hodge_laplacian(transformed) / maximum
            adjusted[name] += config.strength * operator.adjoint(penalty_gradient)
    elif config.method == "architecture_balance":
        weight_gradients, bias_gradients = hidden_balance_gradients(
            [parameters["W1"], parameters["W2"]],
            [parameters["b1"], parameters["b2"]],
        )
        assert bias_gradients is not None
        hidden_width = parameters["W1"].shape[0]
        adjusted["W1"] += config.strength * weight_gradients[0] / hidden_width
        adjusted["W2"] += config.strength * weight_gradients[1] / hidden_width
        adjusted["b1"] += config.strength * bias_gradients[0] / hidden_width

    for name in ("W1", "W2"):
        adjusted[name] += config.weight_decay * parameters[name]

    if config.method == "dstm_sobolev":
        for name in ("W1", "W2"):
            adjusted[name] = boundary_sobolev_filter(
                adjusted[name], mu=config.strength, normalized=True
            )
    elif config.method == "matched_spectrum_sobolev":
        for name in ("W1", "W2"):
            operator = random_operators[name]
            transformed = operator.forward(adjusted[name])
            filtered = boundary_sobolev_filter(
                transformed, mu=config.strength, normalized=True
            )
            adjusted[name] = operator.adjoint(filtered)
    elif config.method == "grid_sobolev":
        for name in ("W1", "W2"):
            adjusted[name] = _grid_sobolev_filter(adjusted[name], config.strength)

    return adjusted


def _mechanism_metrics(parameters: dict[str, np.ndarray]) -> tuple[float, float, float]:
    weight_norm_squared = sum(
        float(np.vdot(parameters[name], parameters[name]).real) for name in ("W1", "W2")
    )
    boundary_norm_squared = sum(
        float(np.vdot(bdry(parameters[name]), bdry(parameters[name])).real)
        for name in ("W1", "W2")
    )
    residuals = hidden_balance_residuals(
        [parameters["W1"], parameters["W2"]],
        [parameters["b1"], parameters["b2"]],
    )
    balance_norm = np.sqrt(
        sum(float(np.vdot(residual, residual).real) for residual in residuals)
    )
    path_energy = path_diamond_energy(parameters["W1"], parameters["W2"])
    return (
        float(np.sqrt(boundary_norm_squared / max(weight_norm_squared, 1e-30))),
        float(balance_norm / max(weight_norm_squared, 1e-30)),
        float(np.sqrt(2.0 * path_energy / max(weight_norm_squared**2, 1e-30))),
    )


def train(
    config: Config,
    seed: int,
    epochs: int,
    data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    *,
    return_parameters: bool = False,
) -> tuple[RunResult, dict[str, np.ndarray] | None]:
    x_train, y_train, x_validation, y_validation, x_test, y_test = data
    parameters = _initialize(seed)
    velocity = {name: np.zeros_like(value) for name, value in parameters.items()}
    operator_rng = np.random.default_rng(90_000 + seed)
    random_operators = {
        name: SignedPermutation(parameters[name].shape, operator_rng) for name in ("W1", "W2")
    }
    validation_history: list[float] = []
    start = time.perf_counter()

    for epoch in range(epochs):
        _, gradients = _loss_and_gradient(parameters, x_train, y_train)
        adjusted = _apply_method(parameters, gradients, config, random_operators)
        progress = epoch / max(epochs - 1, 1)
        schedule = 1.0 if progress < 0.7 else 0.3
        for name in parameters:
            velocity[name] = config.momentum * velocity[name] + adjusted[name]
            parameters[name] -= config.learning_rate * schedule * velocity[name]
        validation_history.append(_metrics(parameters, x_validation, y_validation)["nll"])

    wall_seconds = time.perf_counter() - start
    train_metrics = _metrics(parameters, x_train, y_train)
    validation_metrics = _metrics(parameters, x_validation, y_validation)
    test_metrics = _metrics(parameters, x_test, y_test)
    corruption_rng = np.random.default_rng(70_000 + seed)
    corrupted = np.clip(x_test + corruption_rng.normal(scale=0.15, size=x_test.shape), 0.0, 1.0)
    corrupted_accuracy = _metrics(parameters, corrupted, y_test)["accuracy"]
    boundary_ratio, balance_ratio, path_ratio = _mechanism_metrics(parameters)
    result = RunResult(
        method=config.method,
        seed=seed,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        strength=config.strength,
        epochs=epochs,
        train_nll=train_metrics["nll"],
        validation_nll=validation_metrics["nll"],
        validation_nll_auc=float(np.mean(validation_history)),
        test_nll=test_metrics["nll"],
        test_accuracy=test_metrics["accuracy"],
        test_brier=test_metrics["brier"],
        corrupted_accuracy=corrupted_accuracy,
        boundary_ratio=boundary_ratio,
        balance_ratio=balance_ratio,
        path_diamond_ratio=path_ratio,
        wall_seconds=wall_seconds,
    )
    return result, parameters if return_parameters else None


def _configuration_grid(quick: bool) -> dict[str, list[Config]]:
    learning_rates = (0.15, 0.35) if quick else (0.15, 0.30, 0.45)
    grid: dict[str, list[Config]] = {}
    grid["baseline"] = [
        Config("baseline", learning_rate, weight_decay)
        for learning_rate in learning_rates
        for weight_decay in ((0.0, 1e-4) if not quick else (1e-4,))
    ]
    penalty_strengths = (3e-4, 3e-3)
    filter_strengths = (0.5, 2.0)
    for method in ("dstm_penalty", "matched_spectrum_penalty", "architecture_balance"):
        grid[method] = [
            Config(method, learning_rate, 1e-4, strength)
            for learning_rate in learning_rates
            for strength in penalty_strengths
        ]
    for method in ("dstm_sobolev", "matched_spectrum_sobolev", "grid_sobolev"):
        grid[method] = [
            Config(method, learning_rate, 1e-4, strength)
            for learning_rate in learning_rates
            for strength in filter_strengths
        ]
    return grid


def _load_data():
    x, y = load_digits(return_X_y=True)
    x = x.astype(float) / 16.0
    x_train_validation, x_test, y_train_validation, y_test = train_test_split(
        x, y, test_size=0.2, random_state=1234, stratify=y
    )
    x_train, x_validation, y_train, y_validation = train_test_split(
        x_train_validation,
        y_train_validation,
        test_size=0.2,
        random_state=5678,
        stratify=y_train_validation,
    )
    return x_train, y_train, x_validation, y_validation, x_test, y_test


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _paired_interval(values: np.ndarray, seed: int = 2026) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    resampled = np.mean(
        values[rng.integers(0, len(values), size=(10_000, len(values)))], axis=1
    )
    low, high = np.quantile(resampled, (0.025, 0.975))
    return float(np.mean(values)), float(low), float(high)


def _summarize(results: list[RunResult]) -> dict[str, Any]:
    by_method = {method: sorted((r for r in results if r.method == method), key=lambda r: r.seed) for method in METHODS}
    baseline = by_method["baseline"]
    summary: dict[str, Any] = {}
    for method, runs in by_method.items():
        metrics: dict[str, Any] = {}
        for field in (
            "test_nll",
            "test_accuracy",
            "validation_nll_auc",
            "test_brier",
            "corrupted_accuracy",
            "boundary_ratio",
            "balance_ratio",
            "wall_seconds",
        ):
            values = np.array([getattr(run, field) for run in runs])
            metrics[field] = {
                "mean": float(np.mean(values)),
                "standard_deviation": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            }
        if method != "baseline":
            accuracy_difference = np.array(
                [run.test_accuracy - base.test_accuracy for run, base in zip(runs, baseline)]
            )
            nll_difference = np.array(
                [run.test_nll - base.test_nll for run, base in zip(runs, baseline)]
            )
            metrics["paired_accuracy_difference"] = dict(
                zip(("mean", "ci95_low", "ci95_high"), _paired_interval(accuracy_difference))
            )
            metrics["paired_nll_difference"] = dict(
                zip(("mean", "ci95_low", "ci95_high"), _paired_interval(nll_difference))
            )
        summary[method] = metrics
    return summary


def _specificity_comparisons(results: list[RunResult]) -> dict[str, Any]:
    by_method = {
        method: {run.seed: run for run in results if run.method == method}
        for method in METHODS
    }
    comparisons = (
        ("dstm_penalty", "matched_spectrum_penalty"),
        ("dstm_sobolev", "matched_spectrum_sobolev"),
    )
    output: dict[str, Any] = {}
    for first, second in comparisons:
        common_seeds = sorted(set(by_method[first]) & set(by_method[second]))
        metrics: dict[str, Any] = {}
        for field in ("test_nll", "test_accuracy", "validation_nll_auc"):
            differences = np.array(
                [
                    getattr(by_method[first][seed], field)
                    - getattr(by_method[second][seed], field)
                    for seed in common_seeds
                ]
            )
            metrics[f"{field}_first_minus_second"] = dict(
                zip(("mean", "ci95_low", "ci95_high"), _paired_interval(differences))
            )
        output[f"{first}_versus_{second}"] = metrics
    return output


def _permutation_diagnostics() -> dict[str, Any]:
    rng = np.random.default_rng(20260813)
    parameters = _initialize(31415)
    _, task_gradients = _loss_and_gradient(
        parameters, rng.normal(size=(64, 64)), rng.integers(0, 10, size=64)
    )
    permutation = rng.permutation(parameters["W1"].shape[0])
    inverse = np.argsort(permutation)
    transformed_parameters = {
        "W1": parameters["W1"][permutation],
        "b1": parameters["b1"][permutation],
        "W2": parameters["W2"][:, permutation],
        "b2": parameters["b2"].copy(),
    }
    transformed_gradients = {
        "W1": task_gradients["W1"][permutation],
        "b1": task_gradients["b1"][permutation],
        "W2": task_gradients["W2"][:, permutation],
        "b2": task_gradients["b2"].copy(),
    }

    def undo(values: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {
            "W1": values["W1"][inverse],
            "b1": values["b1"][inverse],
            "W2": values["W2"][:, inverse],
            "b2": values["b2"],
        }

    def relative_error(left: dict[str, np.ndarray], right: dict[str, np.ndarray]) -> float:
        numerator = np.sqrt(sum(np.linalg.norm(left[name] - right[name]) ** 2 for name in left))
        denominator = np.sqrt(sum(np.linalg.norm(left[name]) ** 2 for name in left))
        return float(numerator / max(denominator, 1e-30))

    diagnostics: dict[str, Any] = {}
    operator_seed = 99
    for method in METHODS:
        config = Config(
            method,
            learning_rate=0.3,
            strength=0.003
            if method in ("dstm_penalty", "matched_spectrum_penalty", "architecture_balance")
            else 1.0,
        )
        random_rng = np.random.default_rng(operator_seed)
        operators = {
            name: SignedPermutation(parameters[name].shape, random_rng) for name in ("W1", "W2")
        }
        original_update = _apply_method(parameters, task_gradients, config, operators)
        random_rng = np.random.default_rng(operator_seed)
        transformed_operators = {
            name: SignedPermutation(transformed_parameters[name].shape, random_rng)
            for name in ("W1", "W2")
        }
        transformed_update = _apply_method(
            transformed_parameters, transformed_gradients, config, transformed_operators
        )
        diagnostics[method] = {
            "one_step_equivariance_relative_error": relative_error(
                original_update, undo(transformed_update)
            )
        }

    original_boundary_energy = 0.5 * sum(
        np.linalg.norm(bdry(parameters[name])) ** 2 for name in ("W1", "W2")
    )
    permuted_boundary_energy = 0.5 * sum(
        np.linalg.norm(bdry(transformed_parameters[name])) ** 2 for name in ("W1", "W2")
    )
    original_balance = hidden_balance_energy(
        [parameters["W1"], parameters["W2"]], [parameters["b1"], parameters["b2"]]
    )
    permuted_balance = hidden_balance_energy(
        [transformed_parameters["W1"], transformed_parameters["W2"]],
        [transformed_parameters["b1"], transformed_parameters["b2"]],
    )
    original_path = path_diamond_energy(parameters["W1"], parameters["W2"])
    permuted_path = path_diamond_energy(
        transformed_parameters["W1"], transformed_parameters["W2"]
    )
    diagnostics["invariants"] = {
        "dstm_energy_relative_change": float(
            abs(permuted_boundary_energy - original_boundary_energy)
            / max(original_boundary_energy, 1e-30)
        ),
        "architecture_balance_relative_change": float(
            abs(permuted_balance - original_balance) / max(original_balance, 1e-30)
        ),
        "path_diamond_relative_change": float(
            abs(permuted_path - original_path) / max(original_path, 1e-30)
        ),
    }

    def boundary_energy(values: dict[str, np.ndarray]) -> float:
        return 0.5 * sum(
            np.linalg.norm(bdry(values[name])) ** 2 for name in ("W1", "W2")
        )

    def balance_energy(values: dict[str, np.ndarray]) -> float:
        return hidden_balance_energy(
            [values["W1"], values["W2"]], [values["b1"], values["b2"]]
        )

    def diamond_energy(values: dict[str, np.ndarray]) -> float:
        return path_diamond_energy(values["W1"], values["W2"])

    reference_values = {
        "dstm": boundary_energy(parameters),
        "architecture_balance": balance_energy(parameters),
        "path_diamond": diamond_energy(parameters),
    }
    permutation_values = {name: [] for name in reference_values}
    scaling_values = {name: [] for name in reference_values}
    probe = rng.normal(size=(128, parameters["W1"].shape[1]))
    reference_logits = _forward(parameters, probe)[0]
    maximum_permutation_function_error = 0.0
    maximum_scaling_function_error = 0.0
    for _ in range(100):
        sample_permutation = rng.permutation(parameters["W1"].shape[0])
        permuted = {
            "W1": parameters["W1"][sample_permutation],
            "b1": parameters["b1"][sample_permutation],
            "W2": parameters["W2"][:, sample_permutation],
            "b2": parameters["b2"],
        }
        permutation_values["dstm"].append(boundary_energy(permuted))
        permutation_values["architecture_balance"].append(balance_energy(permuted))
        permutation_values["path_diamond"].append(diamond_energy(permuted))
        maximum_permutation_function_error = max(
            maximum_permutation_function_error,
            float(np.max(np.abs(_forward(permuted, probe)[0] - reference_logits))),
        )

        scale = np.exp(rng.uniform(-2.0, 2.0, size=parameters["W1"].shape[0]))
        scaled = {
            "W1": scale[:, None] * parameters["W1"],
            "b1": scale * parameters["b1"],
            "W2": parameters["W2"] / scale[None, :],
            "b2": parameters["b2"],
        }
        scaling_values["dstm"].append(boundary_energy(scaled))
        scaling_values["architecture_balance"].append(balance_energy(scaled))
        scaling_values["path_diamond"].append(diamond_energy(scaled))
        maximum_scaling_function_error = max(
            maximum_scaling_function_error,
            float(np.max(np.abs(_forward(scaled, probe)[0] - reference_logits))),
        )

    def variation(values: dict[str, list[float]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for name, samples in values.items():
            sample_array = np.asarray(samples)
            reference = reference_values[name]
            result[name] = {
                "coefficient_of_variation": float(
                    np.std(sample_array) / max(abs(np.mean(sample_array)), 1e-30)
                ),
                "maximum_relative_change": float(
                    np.max(np.abs(sample_array - reference)) / max(abs(reference), 1e-30)
                ),
            }
        return result

    diagnostics["one_hundred_hidden_permutations"] = variation(permutation_values)
    diagnostics["one_hundred_positive_gauge_rescalings"] = variation(scaling_values)
    diagnostics["function_preservation_max_absolute_error"] = {
        "permutations": maximum_permutation_function_error,
        "positive_rescalings": maximum_scaling_function_error,
    }
    return diagnostics


def _permutation_twin(
    config: Config,
    data,
    epochs: int = 40,
) -> dict[str, float]:
    x_train, y_train, _, _, x_test, _ = data
    original = _initialize(2718)
    rng = np.random.default_rng(1618)
    permutation = rng.permutation(original["W1"].shape[0])
    twin = {
        "W1": original["W1"][permutation].copy(),
        "b1": original["b1"][permutation].copy(),
        "W2": original["W2"][:, permutation].copy(),
        "b2": original["b2"].copy(),
    }
    velocities = [
        {name: np.zeros_like(value) for name, value in original.items()},
        {name: np.zeros_like(value) for name, value in twin.items()},
    ]
    operator_sets = []
    for _ in range(2):
        operator_rng = np.random.default_rng(92_718)
        operator_sets.append(
            {name: SignedPermutation(original[name].shape, operator_rng) for name in ("W1", "W2")}
        )

    for epoch in range(epochs):
        for parameters, velocity, operators in zip((original, twin), velocities, operator_sets):
            _, gradient = _loss_and_gradient(parameters, x_train, y_train)
            adjusted = _apply_method(parameters, gradient, config, operators)
            schedule = 1.0 if epoch / max(epochs - 1, 1) < 0.7 else 0.3
            for name in parameters:
                velocity[name] = config.momentum * velocity[name] + adjusted[name]
                parameters[name] -= config.learning_rate * schedule * velocity[name]

    first = _softmax(_forward(original, x_test)[0])
    second = _softmax(_forward(twin, x_test)[0])
    return {
        "prediction_relative_rms": float(
            np.linalg.norm(first - second) / max(np.linalg.norm(first), 1e-30)
        ),
        "prediction_argmax_agreement": float(
            np.mean(np.argmax(first, axis=1) == np.argmax(second, axis=1))
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/results/neural_boundary_study"),
    )
    parser.add_argument("--quick", action="store_true", help="Run a small smoke study")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = _load_data()
    grid = _configuration_grid(args.quick)
    tuning_seeds = (0, 1) if args.quick else (0, 1, 2)
    confirmation_seeds = (10, 11, 12) if args.quick else tuple(range(10, 18))
    tuning_epochs = 35 if args.quick else 70
    confirmation_epochs = 55 if args.quick else 100

    tuning_rows: list[dict[str, Any]] = []
    selected: dict[str, Config] = {}
    for method in METHODS:
        candidates = grid[method]
        candidate_scores: list[float] = []
        for candidate_index, config in enumerate(candidates):
            validation_scores: list[float] = []
            for seed in tuning_seeds:
                result, _ = train(config, seed, tuning_epochs, data)
                row = asdict(result)
                row["candidate_index"] = candidate_index
                tuning_rows.append(row)
                validation_scores.append(result.validation_nll)
            candidate_scores.append(float(np.mean(validation_scores)))
        selected[method] = candidates[int(np.argmin(candidate_scores))]

    confirmation: list[RunResult] = []
    for method in METHODS:
        for seed in confirmation_seeds:
            result, _ = train(selected[method], seed, confirmation_epochs, data)
            confirmation.append(result)

    diagnostics = _permutation_diagnostics()
    twins = {
        method: _permutation_twin(selected[method], data) for method in METHODS
    }
    summary = {
        "protocol": {
            "tuning_seeds": tuning_seeds,
            "confirmation_seeds": confirmation_seeds,
            "tuning_epochs": tuning_epochs,
            "confirmation_epochs": confirmation_epochs,
            "selection_metric": "mean validation NLL",
        },
        "selected_configs": {method: asdict(config) for method, config in selected.items()},
        "confirmation": _summarize(confirmation),
        "specificity_comparisons": _specificity_comparisons(confirmation),
        "permutation_diagnostics": diagnostics,
        "permutation_twin_training": twins,
    }
    _write_csv(args.output_dir / "tuning_runs.csv", tuning_rows)
    _write_csv(
        args.output_dir / "confirmation_runs.csv", [asdict(result) for result in confirmation]
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
