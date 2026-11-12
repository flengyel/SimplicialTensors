# Boundary and random weight augmentation

[The script](../../experiments/boundary_augmented_network.py) compares three PyTorch networks on Gaussian regression data. The default widths are 64, 256, 128, 64, 32, and 10. The networks use Leaky ReLU, Adam, a step scheduler, and batches of 32 samples for 40 epochs.

The baseline receives no perturbation. After each optimizer step, the boundary arm computes `bdry_mod1(weight)` on a detached NumPy copy and duplicates its middle index with `degen`. The random arm duplicates a Gaussian array of the same boundary shape. Both arms scale each nonzero update to the same relative Frobenius norm, `1e-5` by default, and apply it to the final three weight matrices. Both arms skip weight matrices with an axis of length one, whose boundary shape is empty. The random generator advances between updates.

```sh
python -m pip install -e .
python -m pip install torch
python experiments/boundary_augmented_network.py --small --epochs 2 --batch-size 4 --seed 123
```

`--small` selects widths 8, 12, 10, 8, 6, and 3. `--epochs`, `--batch-size`, `--seed`, and `--relative-norm` set the remaining parameters. Each arm starts from the same seeded weights and receives the same Gaussian batches. The script prints its settings and training losses. It writes no result file.

Weight mutation follows backpropagation and the optimizer step. These are perturbation experiments on labeled weights. They do not retain the cycle channel or implement a boundary controller, and their training losses provide no validation-accuracy result. See the [controlled neural study](neural_boundary_study.md) for separate empirical comparisons.
