# Initialization and reconstruction losses

[The script](../experiments/consolidated_experiment.py) compares two approximations to a random tensor $T$. Write $\widehat X=X/\|X\|_F$ for nonzero $X$ and $\widehat 0=0$. The boundary error is

$$
\|\widehat T-\widehat{s_0\partial T}\|_F.
$$

For an independently initialized array $S$ with the shape of $\partial T$, the comparison error is

$$
(1+\|\partial S\|_F)\|\widehat T-\widehat{s_0S}\|_F.
$$

Here $s_0$ duplicates index zero along every axis. Both operations use the installed package. These approximations do not recover the cycle component of $T$.

```sh
python -m pip install -e .
python experiments/consolidated_experiment.py --shape 3,3 --trials 6 --resamples 64 --seed 123
```

The default run uses 200 trials for each of `(128, 64)` and `(32, 3, 3, 3)`, with Glorot, He, and orthogonal initialization. Repeat `--shape` for several shapes. A local generator samples new arrays $T$ and $S$ on every trial. Glorot uses the first two axis lengths as its fan parameters. He uses the first axis length. Orthogonal initialization flattens all axes except the first before taking an SVD.

Each result records the shape, initialization, seed, trial count, mean error difference, and two-sided paired permutation p-value. Permutations exchange the two conditions within trials. The test enumerates all exchanges for small samples and otherwise uses seeded resampling. Its null hypothesis is paired-label exchangeability. The output also includes the Kolmogorov–Smirnov distance between the empirical marginal distributions and a mean difference standardized by their pooled marginal standard deviations. It does not report an independence-based KS p-value for the paired observations.
