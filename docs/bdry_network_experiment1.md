# Additive boundary network prototype

[The script](../experiments/bdry_network_experiment1.py) compares two PyTorch multilayer perceptrons on Gaussian regression data. The default widths are 64, 256, 128, 64, 32, and 10. Each network trains with Adam and mean squared error for ten epochs, with batches of 32 samples.

After each optimizer step, the augmented network computes the NumPy array `degen(bdry(weight), 0)` and converts it to the weight's PyTorch device and dtype. For each nonzero update direction, the perturbation norm is `boundary_scale * norm(weight)`. The default relative scale is `1e-3`. Conversion and mutation occur outside autograd, after backpropagation. Weight matrices with an axis of length one have an empty boundary shape and receive no perturbation.

```sh
python -m pip install -e .
python -m pip install torch
python experiments/bdry_network_experiment1.py --small --epochs 2 --batch-size 4 --seed 123
```

The command runs a small network with widths 8, 12, 10, 8, 6, and 3. Omit `--small` for the default widths. `--epochs`, `--batch-size`, `--seed`, and `--boundary-scale` configure the run. The two arms start from the same seeded parameters and receive the same sequence of Gaussian batches. Output records these settings and each training loss. No result file is written.

This experiment applies an additive perturbation to labeled weights. It has no validation set, retained cycle channel, or task-dependent boundary controller. The identity $\partial^2=0$ gives no identity for repeated updates of the same weight shape. The [controlled neural study](neural_boundary_study.md) provides separate comparison experiments.
