# Experiment catalog

Run scripts from an installed checkout unless the linked page gives a
different entry point. Each page describes the code on the default branch,
its dependencies, output, and known execution or interpretation limits.

## Scripts and notebook

- `experiments/Gemini_hallicination.py` — Tensor-product examples comparing selected Moore fillers with their input tensors. See [Gemini_hallicination](Gemini_hallicination.md).
- `experiments/atlas_cospectral_scan.py` — Graph-atlas spectral grouping and mod-two deletion-boundary rank enumeration. See [atlas_cospectral_scan](atlas_cospectral_scan.md).
- `experiments/bdry_network_experiment1.py` — Unseeded PyTorch prototype with additive boundary perturbations. See [bdry_network_experiment1](bdry_network_experiment1.md).
- `experiments/boundary_augmented_network.py` — PyTorch prototype comparing boundary and random weight perturbations. See [boundary_augmented_network](boundary_augmented_network.md).
- `experiments/boundary_observer_study.py` — Controlled forecasting and copied-weight interventions using boundary observations. See [boundary_observer_study](boundary_observer_study.md).
- `experiments/clayworth_simplicial_regression.py` — Finite regression checks for proposed simplicial-object constructions. See [clayworth_simplicial_regression](clayworth_simplicial_regression.md).
- `experiments/consolidated_experiment.py` — Random-matrix comparisons of boundaries, degeneracy, and distribution summaries. See [consolidated_experiment](consolidated_experiment.md).
- `experiments/consolidated_experiment_with_n01.py` — The random-matrix study with an additional Gaussian input family. See [consolidated_experiment_with_n01](consolidated_experiment_with_n01.md).
- `experiments/convex_boundary_study.py` — Validation-tuned matrix regression with boundary and matched penalty controls. See [convex_boundary_study](convex_boundary_study.md).
- `experiments/cospectral_proof.py` — Exhaustive mod-two boundary ranks for an atlas pair and a zero-boundary test for Petersen labelings. See [cospectral_proof](cospectral_proof.md).
- `experiments/count_standard_basis_tensors.py` — Enumeration of coordinate tensors accepted by the degeneracy test. See [count_standard_basis_tensors](count_standard_basis_tensors.md).
- `experiments/degenerate_counterexample.py` — A matrix example separating selected filler recovery from filler uniqueness. See [degenerate_counterexample](degenerate_counterexample.md).
- `experiments/discrepancy_test.py` — Sampled reconstruction comparisons with tabular and plot outputs. See [discrepancy_test](discrepancy_test.md).
- `experiments/examples_range_tensors.py` — Fixed range-tensor horn, degeneracy, and filler comparisons. See [examples_range_tensors](examples_range_tensors.md).
- `experiments/filler_combinatorics.py` — Symbolic comparison of uncovered horn coordinates and Moore filler corrections. See [filler_combinatorics](filler_combinatorics.md).
- `experiments/fixed_index_face_independence.py` — Floating-point rank checks on surviving coordinate images of each face. See [fixed_index_face_independence](fixed_index_face_independence.md).
- `experiments/g115_g117_full_enumeration.py` — Exact mod-two boundary-rank histograms for all 720 labelings of each graph. See [g115_g117_full_enumeration](g115_g117_full_enumeration.md).
- `experiments/homology_basis.py` — Counts of coordinate tensors failing the degeneracy test. See [homology_basis](homology_basis.md).
- `experiments/homotopy_constraint_verification.py` — Comparison of horn-support tuples with the common kernel of included faces. See [homotopy_constraint_verification](homotopy_constraint_verification.md).
- `experiments/homotopy_constraint_verification_with_independence_test.py` — The horn-support diagnostic with separate face-image independence tests. See [homotopy_constraint_verification_with_independence_test](homotopy_constraint_verification_with_independence_test.md).
- `experiments/horn_map_reduce.py` — Dask enumeration of coordinates absent from the included horn faces. See [horn_map_reduce](horn_map_reduce.md).
- `experiments/loss_experiment.py` — Reconstruction-loss distributions over 10,000 random-matrix trials. See [loss_experiment](loss_experiment.md).
- `experiments/matrix_boundary_homotopy_identity.py` — A permutation-boundary diagnostic with a failing action convention. See [matrix_boundary_homotopy_identity](matrix_boundary_homotopy_identity.md).
- `experiments/n_cycle_conjugation.py` — Cyclic relabeling and signed or mod-two boundary comparisons. See [n_cycle_conjugation](n_cycle_conjugation.md).
- `experiments/n_hypergroupoid_conjecture.py` — Shape predicates and comparisons with the algorithm's chosen horn fillers. See [n_hypergroupoid_conjecture](n_hypergroupoid_conjecture.md).
- `experiments/neural_boundary_study.py` — Paired-seed digits models with validation tuning and neural symmetry controls. See [neural_boundary_study](neural_boundary_study.md).
- `experiments/normalized_moore_complex_homology.py` — An exploratory floating-point construction with documented homology limitations. See [normalized_moore_complex_homology](normalized_moore_complex_homology.md).
- `experiments/notebooks/petersen_coset_enumeration.ipynb` — SageMath notebook and script for Petersen permutation cosets. See [petersen_coset_enumeration](petersen_coset_enumeration.md).
- `experiments/notebooks/petersen_coset_enumeration.sage` — SageMath notebook and script for Petersen permutation cosets. See [petersen_coset_enumeration](petersen_coset_enumeration.md).
- `experiments/permutation_statistics.py` — Callable cycle statistics and plots, with a broken command-line entry point. See [permutation_statistics](permutation_statistics.md).
- `experiments/petersen_brd_right_cosets.py` — Exact Petersen deletion-boundary ranks from 30,240 distinct adjacency matrices. See [petersen_brd_right_cosets](petersen_brd_right_cosets.md).
- `experiments/random_zero_one_mask_cocycle_test.py` — A masking diagnostic requiring an external cochain helper. See [random_zero_one_mask_cocycle_test](random_zero_one_mask_cocycle_test.md).
- `experiments/sagemath_compatible_tensor_ops.py` — Demonstration of the package's SageMath or SymPy symbolic backend. See [sagemath_compatible_tensor_ops](sagemath_compatible_tensor_ops.md).
- `experiments/sagemath_normalized_tensor_homology.py` — An exploratory SymPy complex on encountered tensors. See [sagemath_normalized_tensor_homology](sagemath_normalized_tensor_homology.md).
- `experiments/shape_generator.py` — Shape enumeration ordered by dimension, axis count, and offset sum. See [shape_generator](shape_generator.md).
- `experiments/simplicial_permutation_group.py` — Finite checks of cycle deletion under vertex removal. See [simplicial_permutation_group](simplicial_permutation_group.md).
- `experiments/verify_degenerate_preference.py` — Finite checks of the algorithm's degenerate filler selection. See [verify_degenerate_preference](verify_degenerate_preference.md).
- `experiments/verify_injective_face_map.py` — Injectivity checks on coordinate tensors surviving a fixed face. See [verify_injective_face_map](verify_injective_face_map.md).

## Stored studies

- `experiments/results/boundary_observer_study/final_metrics.csv` — Per-trajectory final training and held-out task metrics. See [boundary_observer_study](boundary_observer_study.md).
- `experiments/results/boundary_observer_study/forecast_metrics.csv` — Held-out forecasting results after observer tuning. See [boundary_observer_study](boundary_observer_study.md).
- `experiments/results/boundary_observer_study/shadow_interventions.csv` — Training and validation effects of matched copied-weight interventions. See [boundary_observer_study](boundary_observer_study.md).
- `experiments/results/boundary_observer_study/summary.json` — Configuration, diagnostics, aggregate results, and software versions. See [boundary_observer_study](boundary_observer_study.md).
- `experiments/results/boundary_observer_study/trajectories.csv` — Measurements at the recorded training checkpoints. See [boundary_observer_study](boundary_observer_study.md).
- `experiments/results/convex_boundary_study/signal1.csv` — Per-run results for the indicated signal level. See [convex_boundary_study](convex_boundary_study.md).
- `experiments/results/convex_boundary_study/signal1.json` — Protocol, diagnostics, and summaries for the indicated signal level. See [convex_boundary_study](convex_boundary_study.md).
- `experiments/results/convex_boundary_study/signal2.csv` — Per-run results for the indicated signal level. See [convex_boundary_study](convex_boundary_study.md).
- `experiments/results/convex_boundary_study/signal2.json` — Protocol, diagnostics, and summaries for the indicated signal level. See [convex_boundary_study](convex_boundary_study.md).
- `experiments/results/neural_boundary_study/confirmation_runs.csv` — Paired confirmation runs after validation tuning. See [neural_boundary_study](neural_boundary_study.md).
- `experiments/results/neural_boundary_study/summary.json` — Selected methods, result summaries, and invariance diagnostics. See [neural_boundary_study](neural_boundary_study.md).
- `experiments/results/neural_boundary_study/tuning_runs.csv` — Validation-search results for the evaluated methods. See [neural_boundary_study](neural_boundary_study.md).

## Other artifacts

- `experiments/boundary_observer_report.md` — Protocol, stored results, validity checks, and interpretation of the observer study.
- `experiments/experiment_results.csv` — Stored shape, reconstruction-loss, and test statistics from the loss experiment.
- `experiments/output.txt` — Stored symbolic horn and filler comparisons beginning with shape (3,3).
- `experiments/plots/Training and Validation Accuracy Figure_1.png` — Stored plot. Consult the associated experiment page for the generating workflow.
- `experiments/plots/Training and Validation Accuracy Figure_2.png` — Stored plot. Consult the associated experiment page for the generating workflow.
- `experiments/plots/diagonal_vs_properties.png` — Stored plot. Consult the associated experiment page for the generating workflow.
- `experiments/plots/discrepancy_analysis.png` — Stored plot. Consult the associated experiment page for the generating workflow.
- `experiments/plots/properties_by_size.png` — Stored plot. Consult the associated experiment page for the generating workflow.
- `experiments/plots/properties_vs_similarity.png` — Stored plot. Consult the associated experiment page for the generating workflow.
- `experiments/plots/similarity_distribution.png` — Stored plot. Consult the associated experiment page for the generating workflow.
