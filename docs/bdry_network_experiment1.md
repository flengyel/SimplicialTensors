# Additive boundary network prototype

[The script](../experiments/bdry_network_experiment1.py) compares two PyTorch multilayer perceptrons with widths 64, 256, 128, 64, 32, and 10. Both train with Adam and mean squared error on fresh Gaussian inputs and targets at each epoch. Each run has ten epochs and batches of 32 samples. There is no validation set or recorded accuracy endpoint.

After each optimizer step, the augmented network attempts to add `degen(bdry(weight), 0)` to every weight matrix. This is an additive perturbation of labeled weights.

```sh
python -m pip install -e .
python -m pip install torch
python experiments/bdry_network_experiment1.py
```

The script has no command-line options. It prints the training loss each epoch and writes no result file. The training data and network initialization have no explicit seed in this script.

The call `bdry(param.data)` passes a PyTorch tensor directly to a NumPy-oriented function. Execution depends on that unsupported interoperability and may fail before the augmented comparison completes. The later [controlled neural study](neural_boundary_study.md) uses explicit NumPy arrays and matched controls.

The graded identity `bdry(bdry(W)) == 0` alone gives no identity for repeated same-shape updates `W + degen(bdry(W), 0)`.
