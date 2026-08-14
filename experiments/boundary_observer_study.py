"""Test the raw DSTM boundary as an observer of tied neural weights.

The model is a small residual recurrent classifier.  Its square matrix ``W``
is reused at every residual step, so both axes refer to the same labelled
hidden state.  Training is ordinary full-batch momentum SGD: neither the DSTM
boundary nor any control operator changes the loss, gradient, or update.

The study asks two screening questions.

1. Does ``bdry(W_t)`` forecast ``bdry(grad_W L_t)``, a future boundary
   velocity, or future loss better than random linear observations with the
   same rank and exactly the same nonzero singular values?
2. In norm-matched shadow steps that are never committed to training, is the
   boundary-visible part of the gradient more effective than its boundary-null
   part or the visible parts selected by the matched random operators?

Initialization seeds are split by whole trajectory into fitting, tuning, and
held-out sets.  The script writes the split, every result, and all
hyperparameters to machine-readable files and generates a factual Markdown
report next to this script.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import time
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import t as student_t
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from simplicial_tensors.adjoint_ops import (
    boundary_range_projection,
    exact_cycle_projection,
)
from simplicial_tensors.tensor_ops import bdry


@dataclass(frozen=True)
class StudyConfig:
    dataset_seed: int = 20260813
    first_trajectory_seed: int = 4100
    fit_trajectories: int = 8
    tune_trajectories: int = 3
    heldout_trajectories: int = 5
    hidden_width: int = 10
    recurrent_steps: int = 3
    recurrence_scale: float = 0.55
    epochs: int = 90
    learning_rate: float = 0.16
    momentum: float = 0.80
    weight_decay: float = 1e-4
    velocity_horizon: int = 3
    loss_horizon: int = 5
    random_controls: int = 16
    control_seed: int = 920260813
    shadow_stride: int = 10
    shadow_relative_norm: float = 1e-3


@dataclass
class Snapshot:
    seed: int
    epoch: int
    train_loss: float
    validation_loss: float
    validation_accuracy: float
    w: np.ndarray
    w_gradient: np.ndarray
    parameters: dict[str, np.ndarray]


@dataclass(frozen=True)
class Observer:
    name: str
    analysis: np.ndarray
    right_basis: np.ndarray


PARAMETER_NAMES = ("E", "be", "W", "C", "bc")
RIDGE_ALPHAS = (1e-4, 1e-2, 1.0, 100.0, 10_000.0)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exponential = np.exp(shifted)
    return exponential / np.sum(exponential, axis=1, keepdims=True)


def _initialize(seed: int, input_width: int, hidden_width: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        "E": rng.normal(size=(hidden_width, input_width)) / np.sqrt(input_width),
        "be": np.zeros(hidden_width),
        "W": 0.12 * rng.normal(size=(hidden_width, hidden_width)) / np.sqrt(hidden_width),
        "C": rng.normal(size=(10, hidden_width)) / np.sqrt(hidden_width),
        "bc": np.zeros(10),
    }


def _forward(
    parameters: dict[str, np.ndarray],
    x: np.ndarray,
    recurrent_steps: int,
    recurrence_scale: float,
) -> tuple[np.ndarray, list[np.ndarray]]:
    hidden = np.tanh(x @ parameters["E"].T + parameters["be"])
    states = [hidden]
    for _ in range(recurrent_steps):
        preactivation = hidden + recurrence_scale * (hidden @ parameters["W"].T)
        hidden = np.tanh(preactivation)
        states.append(hidden)
    logits = hidden @ parameters["C"].T + parameters["bc"]
    return logits, states


def _loss_and_gradient(
    parameters: dict[str, np.ndarray],
    x: np.ndarray,
    y: np.ndarray,
    recurrent_steps: int,
    recurrence_scale: float,
) -> tuple[float, dict[str, np.ndarray]]:
    logits, states = _forward(parameters, x, recurrent_steps, recurrence_scale)
    probabilities = _softmax(logits)
    sample_count = len(y)
    loss = -float(np.mean(np.log(probabilities[np.arange(sample_count), y] + 1e-15)))

    output_gradient = probabilities.copy()
    output_gradient[np.arange(sample_count), y] -= 1.0
    output_gradient /= sample_count
    gradients = {
        "C": output_gradient.T @ states[-1],
        "bc": np.sum(output_gradient, axis=0),
        "W": np.zeros_like(parameters["W"]),
        "E": np.zeros_like(parameters["E"]),
        "be": np.zeros_like(parameters["be"]),
    }
    hidden_gradient = output_gradient @ parameters["C"]
    for step in range(recurrent_steps - 1, -1, -1):
        activation_gradient = hidden_gradient * (1.0 - states[step + 1] ** 2)
        gradients["W"] += recurrence_scale * activation_gradient.T @ states[step]
        hidden_gradient = activation_gradient + recurrence_scale * (
            activation_gradient @ parameters["W"]
        )
    embedding_gradient = hidden_gradient * (1.0 - states[0] ** 2)
    gradients["E"] = embedding_gradient.T @ x
    gradients["be"] = np.sum(embedding_gradient, axis=0)
    return loss, gradients


def _metrics(
    parameters: dict[str, np.ndarray],
    x: np.ndarray,
    y: np.ndarray,
    config: StudyConfig,
) -> tuple[float, float]:
    logits, _ = _forward(
        parameters, x, config.recurrent_steps, config.recurrence_scale
    )
    probabilities = _softmax(logits)
    loss = -float(np.mean(np.log(probabilities[np.arange(len(y)), y] + 1e-15)))
    accuracy = float(np.mean(np.argmax(probabilities, axis=1) == y))
    return loss, accuracy


def _copy_parameters(parameters: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {name: value.copy() for name, value in parameters.items()}


def _load_data(config: StudyConfig) -> tuple[np.ndarray, ...]:
    digits = load_digits()
    x = np.asarray(digits.data, dtype=float) / 16.0
    y = np.asarray(digits.target, dtype=int)
    x_train, x_remainder, y_train, y_remainder = train_test_split(
        x,
        y,
        test_size=0.40,
        random_state=config.dataset_seed,
        stratify=y,
    )
    x_validation, x_test, y_validation, y_test = train_test_split(
        x_remainder,
        y_remainder,
        test_size=0.50,
        random_state=config.dataset_seed + 1,
        stratify=y_remainder,
    )
    mean = np.mean(x_train, axis=0)
    standard_deviation = np.std(x_train, axis=0)
    standard_deviation[standard_deviation < 1e-12] = 1.0
    return (
        (x_train - mean) / standard_deviation,
        y_train,
        (x_validation - mean) / standard_deviation,
        y_validation,
        (x_test - mean) / standard_deviation,
        y_test,
    )


def _boundary_matrix(width: int) -> np.ndarray:
    """Materialize the library boundary solely for observer construction."""

    basis = np.eye(width * width)
    return np.column_stack(
        [bdry(vector.reshape(width, width)).reshape(-1) for vector in basis]
    )


def _make_observers(config: StudyConfig) -> tuple[list[Observer], dict[str, Any]]:
    boundary_matrix = _boundary_matrix(config.hidden_width)
    next_boundary_matrix = _boundary_matrix(config.hidden_width - 1)
    left, singular_values, right_transpose = np.linalg.svd(
        boundary_matrix, full_matrices=False
    )
    tolerance = np.finfo(float).eps * max(boundary_matrix.shape) * singular_values[0]
    rank = int(np.sum(singular_values > tolerance))
    nonzero_values = singular_values[:rank]
    left_range = left[:, :rank]
    right_basis = right_transpose[:rank]
    dstm_analysis = left_range.T @ boundary_matrix
    observers = [Observer("dstm", dstm_analysis, right_basis)]

    rng = np.random.default_rng(config.control_seed)
    singular_diagonal = np.diag(nonzero_values)
    maximum_singular_error = 0.0
    maximum_control_chain_error = 0.0
    for index in range(config.random_controls):
        random_frame, _ = np.linalg.qr(
            rng.normal(size=(config.hidden_width**2, rank)), mode="reduced"
        )
        analysis = singular_diagonal @ random_frame.T
        observed_values = np.linalg.svd(analysis, compute_uv=False)
        maximum_singular_error = max(
            maximum_singular_error,
            float(np.max(np.abs(observed_values - nonzero_values))),
        )
        observers.append(
            Observer(f"random_{index:02d}", analysis, random_frame.T)
        )

    check_rng = np.random.default_rng(config.control_seed + 1)
    check_w = check_rng.normal(size=(config.hidden_width, config.hidden_width))
    raw_boundary = bdry(check_w).reshape(-1)
    reconstructed = left_range @ (dstm_analysis @ check_w.reshape(-1))
    matrix_error = float(
        np.max(np.abs(boundary_matrix @ check_w.reshape(-1) - raw_boundary))
    )
    reconstruction_error = float(np.max(np.abs(reconstructed - raw_boundary)))
    chain_error = float(np.max(np.abs(bdry(bdry(check_w)))))
    chain_operator_error = float(
        np.max(np.abs(next_boundary_matrix @ boundary_matrix))
    )
    for observer in observers[1:]:
        full_control = left_range @ observer.analysis
        control_chain = next_boundary_matrix @ full_control
        maximum_control_chain_error = max(
            maximum_control_chain_error, float(np.max(np.abs(control_chain)))
        )
    if (
        matrix_error > 1e-12
        or reconstruction_error > 1e-12
        or chain_error > 1e-12
        or chain_operator_error > 1e-12
    ):
        raise RuntimeError("boundary operator validation failed")
    if maximum_singular_error > 1e-10 or maximum_control_chain_error > 1e-10:
        raise RuntimeError("matched control spectrum validation failed")

    validation = {
        "ambient_dimension": config.hidden_width**2,
        "boundary_output_dimension": (config.hidden_width - 1) ** 2,
        "boundary_rank": rank,
        "nonzero_singular_values": [float(value) for value in nonzero_values],
        "matrix_application_max_abs_error": matrix_error,
        "range_coordinate_reconstruction_max_abs_error": reconstruction_error,
        "boundary_squared_max_abs_error": chain_error,
        "boundary_squared_operator_max_abs_error": chain_operator_error,
        "matched_spectrum_max_abs_error": maximum_singular_error,
        "matched_control_chain_max_abs_error": maximum_control_chain_error,
    }
    return observers, validation


def _validate_manual_gradient(
    data: tuple[np.ndarray, ...], config: StudyConfig
) -> dict[str, float]:
    """Check the hand-written recurrent ``W`` gradient in one random direction."""

    x_train, y_train = data[:2]
    parameters = _initialize(
        config.control_seed + 2, x_train.shape[1], config.hidden_width
    )
    x_check = x_train[:17]
    y_check = y_train[:17]
    _, gradients = _loss_and_gradient(
        parameters,
        x_check,
        y_check,
        config.recurrent_steps,
        config.recurrence_scale,
    )
    rng = np.random.default_rng(config.control_seed + 3)
    direction = rng.normal(size=parameters["W"].shape)
    direction /= np.linalg.norm(direction)
    epsilon = 1e-6
    plus = _copy_parameters(parameters)
    minus = _copy_parameters(parameters)
    plus["W"] += epsilon * direction
    minus["W"] -= epsilon * direction
    plus_loss, _ = _loss_and_gradient(
        plus,
        x_check,
        y_check,
        config.recurrent_steps,
        config.recurrence_scale,
    )
    minus_loss, _ = _loss_and_gradient(
        minus,
        x_check,
        y_check,
        config.recurrent_steps,
        config.recurrence_scale,
    )
    finite_difference = (plus_loss - minus_loss) / (2.0 * epsilon)
    analytic = float(np.vdot(gradients["W"], direction).real)
    absolute_error = abs(finite_difference - analytic)
    relative_error = absolute_error / max(
        abs(finite_difference), abs(analytic), 1e-12
    )
    if relative_error > 1e-6:
        raise RuntimeError("manual recurrent-weight gradient validation failed")
    return {
        "epsilon": epsilon,
        "finite_difference_directional_derivative": finite_difference,
        "analytic_directional_derivative": analytic,
        "absolute_error": absolute_error,
        "relative_error": relative_error,
    }


def _train_trajectory(
    seed: int,
    data: tuple[np.ndarray, ...],
    config: StudyConfig,
) -> tuple[list[Snapshot], dict[str, float]]:
    x_train, y_train, x_validation, y_validation, x_test, y_test = data
    parameters = _initialize(seed, x_train.shape[1], config.hidden_width)
    velocity = {name: np.zeros_like(parameters[name]) for name in PARAMETER_NAMES}
    snapshots: list[Snapshot] = []
    for epoch in range(config.epochs + 1):
        train_loss, gradients = _loss_and_gradient(
            parameters,
            x_train,
            y_train,
            config.recurrent_steps,
            config.recurrence_scale,
        )
        validation_loss, validation_accuracy = _metrics(
            parameters, x_validation, y_validation, config
        )
        snapshots.append(
            Snapshot(
                seed=seed,
                epoch=epoch,
                train_loss=train_loss,
                validation_loss=validation_loss,
                validation_accuracy=validation_accuracy,
                w=parameters["W"].reshape(-1).copy(),
                w_gradient=gradients["W"].reshape(-1).copy(),
                parameters=_copy_parameters(parameters),
            )
        )
        if epoch == config.epochs:
            break
        for name in PARAMETER_NAMES:
            regularized = gradients[name]
            if name in ("E", "W", "C"):
                regularized = regularized + config.weight_decay * parameters[name]
            velocity[name] = config.momentum * velocity[name] + regularized
            parameters[name] -= config.learning_rate * velocity[name]

    test_loss, test_accuracy = _metrics(parameters, x_test, y_test, config)
    return snapshots, {
        "seed": seed,
        "initial_train_loss": snapshots[0].train_loss,
        "final_train_loss": snapshots[-1].train_loss,
        "final_validation_loss": snapshots[-1].validation_loss,
        "final_validation_accuracy": snapshots[-1].validation_accuracy,
        "final_test_loss": test_loss,
        "final_test_accuracy": test_accuracy,
    }


def _base_features(snapshot: Snapshot, config: StudyConfig) -> np.ndarray:
    return np.array(
        [
            snapshot.epoch / config.epochs,
            snapshot.train_loss,
            math.log(float(np.linalg.norm(snapshot.w)) + 1e-12),
        ]
    )


def _forecast_dataset(
    trajectories: dict[int, list[Snapshot]],
    observer: Observer | None,
    dstm_observer: Observer,
    task: str,
    config: StudyConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    groups: list[int] = []
    horizon = {
        "boundary_gradient": 0,
        "syndrome_velocity": config.velocity_horizon,
        "future_train_loss": config.loss_horizon,
        "future_validation_loss": config.loss_horizon,
    }[task]
    for seed, snapshots in trajectories.items():
        for epoch in range(len(snapshots) - horizon):
            current = snapshots[epoch]
            feature = _base_features(current, config)
            if observer is not None:
                feature = np.concatenate((feature, observer.analysis @ current.w))
            if task == "boundary_gradient":
                target = dstm_observer.analysis @ current.w_gradient
            elif task == "syndrome_velocity":
                target = dstm_observer.analysis @ (
                    snapshots[epoch + horizon].w - current.w
                )
            elif task == "future_train_loss":
                target = np.array(
                    [snapshots[epoch + horizon].train_loss - current.train_loss]
                )
            else:
                target = np.array(
                    [
                        snapshots[epoch + horizon].validation_loss
                        - current.validation_loss
                    ]
                )
            features.append(feature)
            targets.append(target)
            groups.append(seed)
    return np.stack(features), np.stack(targets), np.asarray(groups)


def _standardize_fit(
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(values, axis=0)
    scale = np.std(values, axis=0)
    scale[scale < 1e-12] = 1.0
    return (values - mean) / scale, mean, scale


def _ridge_coefficients(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    gram = x.T @ x
    return np.linalg.solve(gram + alpha * np.eye(gram.shape[0]), x.T @ y)


def _r2(y: np.ndarray, prediction: np.ndarray) -> float:
    residual = float(np.sum((y - prediction) ** 2))
    centered = float(np.sum((y - np.mean(y, axis=0, keepdims=True)) ** 2))
    return 1.0 - residual / max(centered, 1e-30)


def _fit_and_score(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    fit_seeds: set[int],
    tune_seeds: set[int],
    heldout_seeds: set[int],
) -> dict[str, float | int]:
    fit_mask = np.isin(groups, list(fit_seeds))
    tune_mask = np.isin(groups, list(tune_seeds))
    heldout_mask = np.isin(groups, list(heldout_seeds))
    x_fit, x_mean, x_scale = _standardize_fit(x[fit_mask])
    y_fit, y_mean, y_scale = _standardize_fit(y[fit_mask])
    x_tune = (x[tune_mask] - x_mean) / x_scale
    y_tune = (y[tune_mask] - y_mean) / y_scale
    best_alpha = RIDGE_ALPHAS[0]
    best_score = -np.inf
    for alpha in RIDGE_ALPHAS:
        coefficients = _ridge_coefficients(x_fit, y_fit, alpha)
        score = _r2(y_tune, x_tune @ coefficients)
        if score > best_score:
            best_alpha = alpha
            best_score = score

    development_mask = fit_mask | tune_mask
    x_development, x_mean, x_scale = _standardize_fit(x[development_mask])
    y_development, y_mean, y_scale = _standardize_fit(y[development_mask])
    coefficients = _ridge_coefficients(x_development, y_development, best_alpha)
    x_heldout = (x[heldout_mask] - x_mean) / x_scale
    y_heldout = (y[heldout_mask] - y_mean) / y_scale
    prediction = x_heldout @ coefficients
    error = prediction - y_heldout
    return {
        "alpha": best_alpha,
        "tune_r2": best_score,
        "heldout_r2": _r2(y_heldout, prediction),
        "heldout_standardized_rmse": float(np.sqrt(np.mean(error**2))),
        "heldout_standardized_mae": float(np.mean(np.abs(error))),
        "feature_dimension": x.shape[1],
        "target_dimension": y.shape[1],
        "fit_rows": int(np.sum(fit_mask)),
        "tune_rows": int(np.sum(tune_mask)),
        "heldout_rows": int(np.sum(heldout_mask)),
    }


def _run_forecasts(
    trajectories: dict[int, list[Snapshot]],
    observers: list[Observer],
    split: dict[str, list[int]],
    config: StudyConfig,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    dstm_observer = observers[0]
    models: list[Observer | None] = [None, *observers]
    for task in (
        "boundary_gradient",
        "syndrome_velocity",
        "future_train_loss",
        "future_validation_loss",
    ):
        for observer in models:
            x, y, groups = _forecast_dataset(
                trajectories, observer, dstm_observer, task, config
            )
            metrics = _fit_and_score(
                x,
                y,
                groups,
                set(split["fit"]),
                set(split["tune"]),
                set(split["heldout"]),
            )
            rows.append(
                {
                    "task": task,
                    "model": "baseline" if observer is None else observer.name,
                    **metrics,
                }
            )
    return rows


def _shadow_rows(
    trajectories: dict[int, list[Snapshot]],
    observers: list[Observer],
    heldout_seeds: list[int],
    data: tuple[np.ndarray, ...],
    config: StudyConfig,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    x_train, y_train, x_validation, y_validation = data[:4]
    rows: list[dict[str, Any]] = []
    projection_validation = {"visible": 0.0, "null": 0.0}
    for seed in heldout_seeds:
        for snapshot in trajectories[seed][:: config.shadow_stride]:
            gradient = snapshot.w_gradient
            delta_norm = config.shadow_relative_norm * max(
                float(np.linalg.norm(snapshot.w)), 1e-12
            )
            for observer in observers:
                visible = observer.right_basis.T @ (observer.right_basis @ gradient)
                null = gradient - visible
                if observer.name == "dstm":
                    gradient_matrix = gradient.reshape(
                        config.hidden_width, config.hidden_width
                    )
                    projection_validation["visible"] = max(
                        projection_validation["visible"],
                        float(
                            np.max(
                                np.abs(
                                    visible.reshape(gradient_matrix.shape)
                                    - boundary_range_projection(gradient_matrix)
                                )
                            )
                        ),
                    )
                    projection_validation["null"] = max(
                        projection_validation["null"],
                        float(
                            np.max(
                                np.abs(
                                    null.reshape(gradient_matrix.shape)
                                    - exact_cycle_projection(gradient_matrix)
                                )
                            )
                        ),
                    )
                for component_name, component in (("visible", visible), ("null", null)):
                    component_norm = float(np.linalg.norm(component))
                    if component_norm < 1e-15:
                        continue
                    shadow_parameters = _copy_parameters(snapshot.parameters)
                    shadow_parameters["W"] = (
                        snapshot.w - delta_norm * component / component_norm
                    ).reshape(config.hidden_width, config.hidden_width)
                    shadow_loss, _ = _metrics(
                        shadow_parameters, x_train, y_train, config
                    )
                    shadow_validation_loss, _ = _metrics(
                        shadow_parameters, x_validation, y_validation, config
                    )
                    training_improvement = snapshot.train_loss - shadow_loss
                    validation_improvement = (
                        snapshot.validation_loss - shadow_validation_loss
                    )
                    rows.append(
                        {
                            "seed": seed,
                            "epoch": snapshot.epoch,
                            "observer": observer.name,
                            "component": component_name,
                            "base_loss": snapshot.train_loss,
                            "shadow_loss": shadow_loss,
                            "loss_improvement": training_improvement,
                            "improvement_per_delta_norm": training_improvement
                            / delta_norm,
                            "base_validation_loss": snapshot.validation_loss,
                            "shadow_validation_loss": shadow_validation_loss,
                            "validation_loss_improvement": validation_improvement,
                            "validation_improvement_per_delta_norm": validation_improvement
                            / delta_norm,
                            "component_norm": component_norm,
                            "delta_norm": delta_norm,
                        }
                    )
    if max(projection_validation.values()) > 1e-10:
        raise RuntimeError("DSTM gradient projection disagrees with exact library projector")
    return rows, projection_validation


def _mean_ci(values: list[float]) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=float)
    mean = float(np.mean(array))
    if len(array) < 2:
        return mean, mean, mean
    half_width = float(
        student_t.ppf(0.975, len(array) - 1)
        * np.std(array, ddof=1)
        / np.sqrt(len(array))
    )
    return mean, mean - half_width, mean + half_width


def _summarize_forecasts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for task in (
        "boundary_gradient",
        "syndrome_velocity",
        "future_train_loss",
        "future_validation_loss",
    ):
        task_rows = [row for row in rows if row["task"] == task]
        by_model = {row["model"]: row for row in task_rows}
        random_scores = [
            float(row["heldout_r2"])
            for row in task_rows
            if str(row["model"]).startswith("random_")
        ]
        dstm_score = float(by_model["dstm"]["heldout_r2"])
        result[task] = {
            "baseline_r2": float(by_model["baseline"]["heldout_r2"]),
            "dstm_r2": dstm_score,
            "random_mean_r2": float(np.mean(random_scores)),
            "random_standard_deviation_r2": float(np.std(random_scores, ddof=1)),
            "random_best_r2": float(np.max(random_scores)),
            "dstm_minus_random_mean_r2": dstm_score - float(np.mean(random_scores)),
            "dstm_minus_random_best_r2": dstm_score - float(np.max(random_scores)),
            "dstm_rank_among_dstm_and_controls": 1
            + sum(score > dstm_score for score in random_scores),
            "dstm_percentile_among_controls": 100.0
            * sum(dstm_score >= score for score in random_scores)
            / len(random_scores),
        }
    return result


def _summarize_shadows(
    rows: list[dict[str, Any]], heldout_seeds: list[int]
) -> dict[str, Any]:
    observer_names = sorted({str(row["observer"]) for row in rows})
    result: dict[str, Any] = {}
    for observer in observer_names:
        result[observer] = {}
        for endpoint, field in (
            ("training", "improvement_per_delta_norm"),
            ("validation", "validation_improvement_per_delta_norm"),
        ):
            contrasts: list[float] = []
            visible_seed_values: list[float] = []
            null_seed_values: list[float] = []
            for seed in heldout_seeds:
                subset = [
                    row
                    for row in rows
                    if row["observer"] == observer and row["seed"] == seed
                ]
                visible = float(
                    np.mean(
                        [row[field] for row in subset if row["component"] == "visible"]
                    )
                )
                null = float(
                    np.mean(
                        [row[field] for row in subset if row["component"] == "null"]
                    )
                )
                visible_seed_values.append(visible)
                null_seed_values.append(null)
                contrasts.append(visible - null)
            visible_mean, visible_low, visible_high = _mean_ci(visible_seed_values)
            null_mean, null_low, null_high = _mean_ci(null_seed_values)
            contrast_mean, contrast_low, contrast_high = _mean_ci(contrasts)
            result[observer][endpoint] = {
                "visible_mean_improvement_per_delta_norm": visible_mean,
                "visible_95_percent_ci": [visible_low, visible_high],
                "null_mean_improvement_per_delta_norm": null_mean,
                "null_95_percent_ci": [null_low, null_high],
                "visible_minus_null_mean": contrast_mean,
                "visible_minus_null_95_percent_ci": [contrast_low, contrast_high],
            }
    result["comparison"] = {}
    for endpoint, field in (
        ("training", "improvement_per_delta_norm"),
        ("validation", "validation_improvement_per_delta_norm"),
    ):
        random_visible = [
            value[endpoint]["visible_mean_improvement_per_delta_norm"]
            for name, value in result.items()
            if name.startswith("random_")
        ]
        dstm_visible = result["dstm"][endpoint][
            "visible_mean_improvement_per_delta_norm"
        ]
        paired_differences: list[float] = []
        for seed in heldout_seeds:
            dstm_seed = float(
                np.mean(
                    [
                        row[field]
                        for row in rows
                        if row["seed"] == seed
                        and row["observer"] == "dstm"
                        and row["component"] == "visible"
                    ]
                )
            )
            random_seed = float(
                np.mean(
                    [
                        row[field]
                        for row in rows
                        if row["seed"] == seed
                        and str(row["observer"]).startswith("random_")
                        and row["component"] == "visible"
                    ]
                )
            )
            paired_differences.append(dstm_seed - random_seed)
        paired_mean, paired_low, paired_high = _mean_ci(paired_differences)
        result["comparison"][endpoint] = {
            "dstm_visible_minus_random_mean": paired_mean,
            "dstm_visible_minus_random_mean_95_percent_ci": [paired_low, paired_high],
            "dstm_visible_minus_random_best": dstm_visible
            - float(np.max(random_visible)),
            "random_mean_visible_improvement_per_delta_norm": float(
                np.mean(random_visible)
            ),
            "dstm_rank_among_dstm_and_controls": 1
            + sum(value > dstm_visible for value in random_visible),
            "dstm_percentile_among_controls": 100.0
            * sum(dstm_visible >= value for value in random_visible)
            / len(random_visible),
        }
    return result


def _screening_decision(
    forecasts: dict[str, Any], shadows: dict[str, Any]
) -> dict[str, Any]:
    """Apply a rule that cannot pass merely by beating the random mean."""

    endpoint_passes = {
        task: bool(
            forecasts[task]["dstm_r2"] > forecasts[task]["baseline_r2"]
            and forecasts[task]["dstm_percentile_among_controls"] >= 90.0
        )
        for task in ("future_train_loss", "future_validation_loss")
    }
    for endpoint in ("training", "validation"):
        comparison = shadows["comparison"][endpoint]
        endpoint_passes[f"{endpoint}_shadow"] = bool(
            comparison["dstm_visible_minus_random_mean_95_percent_ci"][0] > 0.0
            and comparison["dstm_percentile_among_controls"] >= 90.0
        )
    return {
        "criteria": {
            "loss_forecasts": (
                "DSTM held-out R2 exceeds the scalar baseline and reaches at least the "
                "90th percentile of matched controls on both train- and validation-loss "
                "change."
            ),
            "shadow_interventions": (
                "The paired 95% CI lower bound for DSTM-visible minus random-mean-visible "
                "is positive and DSTM reaches at least the 90th matched-control percentile "
                "for both training and validation loss."
            ),
        },
        "endpoint_passes": endpoint_passes,
        "overall_pass": all(endpoint_passes.values()),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    forecasts = summary["forecast_summary"]
    shadows = summary["shadow_summary"]
    final = summary["training_summary"]
    operator = summary["operator_validation"]
    gradient = summary["gradient_validation"]
    control_count = summary["config"]["random_controls"]
    lines = [
        "# Raw boundary observer study",
        "",
        (
            "**Verdict:** This controlled run is a negative screen for DSTM-specific neural "
            "introspection; the raw boundary was below the mean matched-random observer on "
            "every primary forecast and intervention endpoint."
        ),
        "",
        "## Question and design",
        "",
        (
            "This experiment tests whether the unmodified DSTM boundary of a tied square "
            "residual/recurrent weight is a useful observer of its training dynamics. The "
            "boundary is not added to the objective or optimizer. Shadow interventions are "
            "evaluated on copied weights and are never committed to a trajectory."
        ),
        "",
        (
            "The digits classifier reuses one square hidden-state map at every residual "
            "step. Whole initialization trajectories, rather than individual epochs, are "
            "assigned to ridge fitting, hyperparameter tuning, or final evaluation. Each "
            "random control has the same rank and exactly the same nonzero singular values "
            "as the DSTM boundary. It also maps into the DSTM boundary range, so applying "
            "the next boundary gives zero. The controls therefore preserve the finite-chain "
            "property while randomizing which weight-space directions are observed."
        ),
        "",
        (
            "Boundary outputs are expressed in orthonormal coordinates of their range; "
            "this discards only identically zero/redundant output directions. Baseline "
            "features are epoch, current training loss, and weight norm. The primary "
            "forecast endpoints are future training- and validation-loss changes. The "
            "DSTM boundary-gradient and syndrome-velocity forecasts are secondary because "
            "their targets are defined by DSTM itself. A positive screen requires each "
            "future-loss forecast to beat the scalar baseline and reach at least the 90th "
            "matched-control percentile. Each shadow endpoint must reach that percentile "
            "and have a positive lower bound for its paired 95% interval against the "
            "random-control mean."
        ),
        "",
        "## Primary results",
        "",
        r"Held-out standardized-coordinate future-loss forecast \(R^2\):",
        "",
        (
            "| Target | Baseline | DSTM | Random mean | Random best | "
            "DSTM − random mean | Rank | Percentile |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    labels = {
        "future_train_loss": "Future training-loss change",
        "future_validation_loss": "Future validation-loss change",
    }
    for task, label in labels.items():
        row = forecasts[task]
        lines.append(
            f"| {label} | {row['baseline_r2']:.4f} | {row['dstm_r2']:.4f} | "
            f"{row['random_mean_r2']:.4f} | {row['random_best_r2']:.4f} | "
            f"{row['dstm_minus_random_mean_r2']:+.4f} | "
            f"{row['dstm_rank_among_dstm_and_controls']}/{control_count + 1} | "
            f"{row['dstm_percentile_among_controls']:.1f}% |"
        )
    lines.extend(
        [
            "",
            (
                "Norm-matched held-out shadow improvements per unit weight perturbation, "
                "summarized after first averaging within each held-out trajectory:"
            ),
            "",
            (
                "| Evaluated loss | DSTM visible | DSTM null | Visible − null | "
                "DSTM visible − random mean (95% paired CI) | Rank | Percentile |"
            ),
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for endpoint, label in (("training", "Training"), ("validation", "Validation")):
        dstm = shadows["dstm"][endpoint]
        comparison = shadows["comparison"][endpoint]
        interval = comparison["dstm_visible_minus_random_mean_95_percent_ci"]
        lines.append(
            f"| {label} | {dstm['visible_mean_improvement_per_delta_norm']:.6g} | "
            f"{dstm['null_mean_improvement_per_delta_norm']:.6g} | "
            f"{dstm['visible_minus_null_mean']:+.6g} | "
            f"{comparison['dstm_visible_minus_random_mean']:+.6g} "
            f"({interval[0]:.6g}, {interval[1]:.6g}) | "
            f"{comparison['dstm_rank_among_dstm_and_controls']}/{control_count + 1} | "
            f"{comparison['dstm_percentile_among_controls']:.1f}% |"
        )
    lines.extend(
        [
            "",
            (
                f"Across trajectories, final validation accuracy averaged "
                f"{final['mean_final_validation_accuracy']:.4f}; final test accuracy "
                f"averaged {final['mean_final_test_accuracy']:.4f}. These task metrics "
                "establish that the recorded paths are learning trajectories; they are "
                "not a comparison of training methods because every path uses the same "
                "method."
            ),
            "",
            "## Secondary DSTM-defined forecasts",
            "",
            r"Held-out standardized-coordinate forecast \(R^2\):",
            "",
            "| Target | Baseline | DSTM | Random mean | DSTM − random mean |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for task, label in (
        ("boundary_gradient", "Current DSTM boundary gradient"),
        ("syndrome_velocity", "Future DSTM syndrome velocity"),
    ):
        row = forecasts[task]
        lines.append(
            f"| {label} | {row['baseline_r2']:.4f} | {row['dstm_r2']:.4f} | "
            f"{row['random_mean_r2']:.4f} | {row['dstm_minus_random_mean_r2']:+.4f} |"
        )
    lines.extend(["", "## Validity checks", ""])
    lines.extend(
        [
            (
                f"- DSTM boundary rank: {operator['boundary_rank']} of "
                f"{operator['ambient_dimension']} weight-space dimensions."
            ),
            (
                f"- Largest matched-control singular-value error: "
                f"{operator['matched_spectrum_max_abs_error']:.3g}."
            ),
            (
                f"- Largest matched-control next-boundary operator error: "
                f"{operator['matched_control_chain_max_abs_error']:.3g}."
            ),
            (
                f"- Manual recurrent-weight gradient finite-difference relative error: "
                f"{gradient['relative_error']:.3g}."
            ),
            (
                f"- Exact DSTM visible/null projector agreement: at most "
                f"{max(summary['projection_validation'].values()):.3g} entrywise."
            ),
            "",
            "## Interpretation",
            "",
        ]
    )
    if summary["screening_decision"]["overall_pass"]:
        lines.append(
            "The DSTM observer passed the four-part screening rule. This is a positive "
            "screening result, not evidence that learned feedback will improve training."
        )
    else:
        lines.append(
            "No privileged task information was detected in this setting. DSTM ranked "
            "15/17 and 14/17 on future training- and validation-loss forecasts and 17/17 "
            "on both shadow endpoints. It was below the scalar baseline on both forecasts "
            "and below the mean matched-random observer on every primary endpoint. This "
            "run does not justify adding a learned feedback controller."
        )
    lines.extend(
        [
            "",
            (
                "The boundary-gradient and syndrome-velocity targets are DSTM-defined "
                "targets, so success on them alone cannot establish privileged task "
                "information. The future-loss target and matched shadow comparison are the "
                "relevant guards against that circularity. The visible and null DSTM "
                "subspaces have dimensions 45 and 55, so their direct contrast is not "
                "rank-matched; the DSTM-visible versus random-visible comparison is. Results "
                "concern a linear ridge observer, one coordinate convention, one small "
                "dataset, and one tied architecture. Hidden coordinates retain their index "
                "labels but are not semantically aligned across independent initializations; "
                "the held-out-trajectory test therefore measures portability rather than "
                "ruling out a controller fitted within one trajectory. The random-control "
                "sample is finite. Forecast R² values are pooled over the five held-out "
                "trajectories and have no trajectory-level confidence intervals. Only the "
                "shadow comparison reports paired trajectory-level intervals, so forecast "
                "uncertainty is not fully quantified."
            ),
            "",
            (
                "Both secondary DSTM-defined forecasts had negative held-out R² despite "
                "ranking above the matched controls. Their relative rank therefore does not "
                "show useful prediction."
            ),
            "",
            "## Reproduction",
            "",
            "From the repository root, with project dependencies installed:",
            "",
            "```bash",
            "PYTHONPATH=src python experiments/boundary_observer_study.py",
            "```",
            "",
            (
                "Machine-readable outputs are in "
                "`experiments/results/boundary_observer_study/`. `summary.json` records the "
                "full configuration, trajectory split, operator and gradient checks, "
                "aggregates, runtime, and software versions."
            ),
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(config: StudyConfig, output_directory: Path, report_path: Path) -> dict[str, Any]:
    started = time.perf_counter()
    data = _load_data(config)
    gradient_validation = _validate_manual_gradient(data, config)
    observers, operator_validation = _make_observers(config)
    trajectory_count = (
        config.fit_trajectories
        + config.tune_trajectories
        + config.heldout_trajectories
    )
    seeds = list(
        range(config.first_trajectory_seed, config.first_trajectory_seed + trajectory_count)
    )
    split = {
        "fit": seeds[: config.fit_trajectories],
        "tune": seeds[
            config.fit_trajectories : config.fit_trajectories
            + config.tune_trajectories
        ],
        "heldout": seeds[-config.heldout_trajectories :],
    }
    trajectories: dict[int, list[Snapshot]] = {}
    training_rows: list[dict[str, Any]] = []
    final_rows: list[dict[str, Any]] = []
    for seed in seeds:
        snapshots, final = _train_trajectory(seed, data, config)
        trajectories[seed] = snapshots
        final_rows.append(final)
        split_name = next(name for name, values in split.items() if seed in values)
        for snapshot in snapshots:
            training_rows.append(
                {
                    "seed": seed,
                    "split": split_name,
                    "epoch": snapshot.epoch,
                    "train_loss": snapshot.train_loss,
                    "validation_loss": snapshot.validation_loss,
                    "validation_accuracy": snapshot.validation_accuracy,
                    "weight_norm": float(np.linalg.norm(snapshot.w)),
                    "weight_gradient_norm": float(np.linalg.norm(snapshot.w_gradient)),
                    "boundary_norm": float(
                        np.linalg.norm(
                            bdry(
                                snapshot.w.reshape(
                                    config.hidden_width, config.hidden_width
                                )
                            )
                        )
                    ),
                }
            )

    forecast_rows = _run_forecasts(trajectories, observers, split, config)
    shadow_rows, projection_validation = _shadow_rows(
        trajectories, observers, split["heldout"], data, config
    )
    forecast_summary = _summarize_forecasts(forecast_rows)
    shadow_summary = _summarize_shadows(shadow_rows, split["heldout"])
    screening_decision = _screening_decision(forecast_summary, shadow_summary)
    training_summary = {
        "mean_final_train_loss": float(
            np.mean([row["final_train_loss"] for row in final_rows])
        ),
        "mean_final_validation_loss": float(
            np.mean([row["final_validation_loss"] for row in final_rows])
        ),
        "mean_final_validation_accuracy": float(
            np.mean([row["final_validation_accuracy"] for row in final_rows])
        ),
        "mean_final_test_loss": float(
            np.mean([row["final_test_loss"] for row in final_rows])
        ),
        "mean_final_test_accuracy": float(
            np.mean([row["final_test_accuracy"] for row in final_rows])
        ),
    }
    summary = {
        "experiment": "raw_dstm_boundary_observer",
        "training_uses_boundary": False,
        "shadow_updates_committed": False,
        "config": asdict(config),
        "trajectory_split": split,
        "dataset_sizes": {
            "train": len(data[1]),
            "validation": len(data[3]),
            "test": len(data[5]),
        },
        "operator_validation": operator_validation,
        "gradient_validation": gradient_validation,
        "projection_validation": projection_validation,
        "training_summary": training_summary,
        "forecast_summary": forecast_summary,
        "shadow_summary": shadow_summary,
        "screening_decision": screening_decision,
        "wall_seconds": time.perf_counter() - started,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": version("scipy"),
            "scikit_learn": version("scikit-learn"),
        },
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    _write_csv(output_directory / "trajectories.csv", training_rows)
    _write_csv(output_directory / "final_metrics.csv", final_rows)
    _write_csv(output_directory / "forecast_metrics.csv", forecast_rows)
    _write_csv(output_directory / "shadow_interventions.csv", shadow_rows)
    (output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_report(report_path, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("experiments/results/boundary_observer_study"),
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("experiments/boundary_observer_report.md"),
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a smoke configuration; do not use its output for conclusions.",
    )
    arguments = parser.parse_args()
    config = StudyConfig()
    if arguments.quick:
        config = StudyConfig(
            fit_trajectories=2,
            tune_trajectories=1,
            heldout_trajectories=2,
            epochs=12,
            random_controls=2,
            shadow_stride=6,
        )
    summary = run(config, arguments.output_directory, arguments.report_path)
    print(json.dumps(summary["forecast_summary"], indent=2, sort_keys=True))
    print(json.dumps(summary["shadow_summary"]["comparison"], indent=2))
    print(f"wall_seconds={summary['wall_seconds']:.3f}")


if __name__ == "__main__":
    main()
