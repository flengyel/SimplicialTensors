"""Run the reconstruction comparison with an additional normal initialization."""

if __package__:
    from .consolidated_experiment import (
        calculate_effect_size, compute_loss, compute_penalty, glorot_init, he_init,
        initialize_weights, ks_test, n01_init, orthogonal_init, permutation_test,
        random_real_tensor, random_tensor, run_experiment_with_initialization,
        main as _main,
    )
else:  # Direct execution from the experiments directory.
    from consolidated_experiment import (
        calculate_effect_size, compute_loss, compute_penalty, glorot_init, he_init,
        initialize_weights, ks_test, n01_init, orthogonal_init, permutation_test,
        random_real_tensor, random_tensor, run_experiment_with_initialization,
        main as _main,
    )

initialization_methods = ['glorot', 'he', 'orthogonal', 'n01']
selected_tensor_shapes_final = [(512, 256), (128, 64), (32, 3, 3, 3)]


def main(argv=None):
    return _main(argv, methods=initialization_methods, shapes=selected_tensor_shapes_final)


if __name__ == '__main__':
    for result in main():
        print(result)
