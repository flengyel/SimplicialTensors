"""Compare paired reconstruction losses under seeded weight initializations."""

import argparse
from typing import Tuple

import numpy as np
from scipy import stats

from simplicial_tensors.tensor_ops import bdry, degen


def random_tensor(shape: Tuple[int, ...], low=1, high=10, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    return rng.integers(low, high, size=shape)


def random_real_tensor(shape: Tuple[int, ...], mean=0.0, std=1.0, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    return rng.normal(mean, std, size=shape)


def compute_penalty(S):
    return 1 + np.linalg.norm(bdry(S))


def compute_loss(T, S):
    """Return the Frobenius distance after normalizing each nonzero array."""
    T_norm = T / np.linalg.norm(T) if np.linalg.norm(T) != 0 else T
    S_norm = S / np.linalg.norm(S) if np.linalg.norm(S) != 0 else S
    return float(np.linalg.norm(T_norm - S_norm))


def glorot_init(shape, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    fan_in, fan_out = shape[0], shape[1]
    limit = np.sqrt(6 / (fan_in + fan_out))
    return rng.uniform(-limit, limit, size=shape)


def he_init(shape, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    return rng.normal(0, np.sqrt(2 / shape[0]), size=shape)


def orthogonal_init(shape, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    flat_shape = (shape[0], int(np.prod(shape[1:])))
    u, _, v = np.linalg.svd(rng.normal(0, 1, flat_shape), full_matrices=False)
    return (u if u.shape == flat_shape else v).reshape(shape)


def n01_init(shape, rng=None):
    return random_real_tensor(shape, rng=rng)


def initialize_weights(shape, method='glorot', rng=None):
    methods = {'glorot': glorot_init, 'he': he_init,
               'orthogonal': orthogonal_init, 'n01': n01_init}
    if method not in methods:
        raise ValueError(f'Unknown initialization method: {method}')
    return methods[method](shape, rng=rng)


def permutation_test(losses_S_adjusted, losses_boundary, *, seed=123, n_resamples=9999):
    """Test the paired mean difference by swapping conditions within trials.

    The two-sided p-value uses exact enumeration when possible and otherwise
    seeded Monte Carlo resampling with SciPy's add-one correction.
    """
    x, y = np.asarray(losses_S_adjusted), np.asarray(losses_boundary)
    if x.ndim != 1 or x.shape != y.shape or x.size < 2:
        raise ValueError('Paired one-dimensional samples of length at least two are required')
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError('Samples must be finite')
    if n_resamples < 1:
        raise ValueError('n_resamples must be positive')
    result = stats.permutation_test(
        (x, y), lambda a, b, axis: np.mean(a - b, axis=axis),
        permutation_type='samples', vectorized=True, alternative='two-sided',
        n_resamples=n_resamples, random_state=np.random.default_rng(seed), batch=256,
    )
    return float(result.statistic), float(result.pvalue)


def ks_test(losses_S_adjusted, losses_boundary):
    """Return a descriptive two-sample KS comparison of the marginal losses."""
    return stats.ks_2samp(losses_S_adjusted, losses_boundary)


def calculate_effect_size(losses_S_adjusted, losses_boundary):
    mean_diff = np.mean(losses_S_adjusted) - np.mean(losses_boundary)
    pooled_std = np.sqrt((np.std(losses_S_adjusted) ** 2 + np.std(losses_boundary) ** 2) / 2)
    if pooled_std == 0:
        return 0.0 if mean_diff == 0 else float(np.copysign(np.inf, mean_diff))
    return float(mean_diff / pooled_std)


def run_experiment_with_initialization(tensor_shape: Tuple[int, ...], init_method: str,
                                      num_trials=100, *, seed=123, n_resamples=9999):
    if num_trials < 2 or len(tensor_shape) < 2 or min(tensor_shape) < 2:
        raise ValueError('Use at least two trials and at least two axes of length two or greater')
    rng = np.random.default_rng(seed)
    losses_S_adjusted, losses_boundary = [], []
    for _ in range(num_trials):
        T = initialize_weights(tensor_shape, method=init_method, rng=rng)
        bdry_T = bdry(T)
        S = initialize_weights(bdry_T.shape, method=init_method, rng=rng)
        losses_S_adjusted.append(compute_loss(T, degen(S, 0)) * compute_penalty(S))
        losses_boundary.append(compute_loss(T, degen(bdry_T, 0)))
    observed_diff, perm_p_value = permutation_test(
        losses_S_adjusted, losses_boundary, seed=seed, n_resamples=n_resamples)
    ks_stat, _ = ks_test(losses_S_adjusted, losses_boundary)
    return {
        'Tensor Shape': tensor_shape, 'Initialization Method': init_method,
        'Seed': seed, 'Trials': num_trials, 'Permutation Resamples': n_resamples,
        'Permutation Test': {'Observed Difference': observed_diff, 'p-value': perm_p_value},
        'KS Distance': float(ks_stat),
        "Cohen's d": calculate_effect_size(losses_S_adjusted, losses_boundary),
    }


initialization_methods = ['glorot', 'he', 'orthogonal']
selected_tensor_shapes_final = [(128, 64), (32, 3, 3, 3)]


def main(argv=None, *, methods=None, shapes=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trials', type=int, default=200)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--resamples', type=int, default=9999)
    parser.add_argument('--shape', action='append', help='Comma-separated axis lengths; repeat for several shapes')
    args = parser.parse_args(argv)
    methods = initialization_methods if methods is None else methods
    shapes = selected_tensor_shapes_final if shapes is None else shapes
    if args.shape:
        shapes = [tuple(int(n) for n in shape.split(',')) for shape in args.shape]
    return [run_experiment_with_initialization(
        shape, method, args.trials, seed=args.seed, n_resamples=args.resamples)
        for method in methods for shape in shapes]


if __name__ == '__main__':
    for result in main():
        print(result)
