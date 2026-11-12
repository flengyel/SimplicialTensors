# Boundary and random weight augmentation prototype

[The script](../experiments/boundary_augmented_network.py) trains three PyTorch networks with widths 64, 256, 128, 64, 32, and 10 using Leaky ReLU, Adam, a step scheduler, and Gaussian inputs and targets. It compares a baseline with boundary and random perturbations of the final three weight matrices. Each run has 40 epochs and batches of 32 samples.

The boundary arm computes `bdry_mod1(weight)` on a NumPy copy, duplicates the middle index with `degen`, converts the result back to PyTorch, and adds it with scale `1e-5`. The random arm duplicates a normal array and adds it with scale `7e-3`. Its local random generator is reset to seed 42 for each layer update.

```sh
python -m pip install -e .
python -m pip install torch
python experiments/boundary_augmented_network.py
```

There are no command-line options or saved result files. The output is a sequence of training losses and any boundary-integration errors.

The training loop mutates weights after the forward pass and before `loss.backward()`. PyTorch may reject this update ordering or compute gradients for inconsistent parameter values. The perturbation scales also differ, so the two augmentation arms are not matched by perturbation norm. Treat the file as a legacy prototype. The [controlled neural study](neural_boundary_study.md) specifies paired runs and matched spectral controls.
