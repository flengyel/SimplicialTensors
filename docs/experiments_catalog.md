# Experiment catalog

Run scripts from an installed checkout unless the linked page gives a
different entry point. Each page describes the code on the default branch,
its dependencies, output, and known execution or interpretation limits.

## Scripts and notebooks

- `experiments/Gemini_hallicination.py` — Tensor-product examples comparing selected Moore fillers with their input tensors. See [Gemini_hallicination](experiments/Gemini_hallicination.md).
- `experiments/atlas_cospectral_scan.py` — Graph-atlas spectral grouping and mod-two deletion-boundary rank enumeration. See [atlas_cospectral_scan](experiments/atlas_cospectral_scan.md).
- `experiments/bdry_network_experiment1.py` — Seeded PyTorch training with boundary perturbations after optimizer steps. See [bdry_network_experiment1](experiments/bdry_network_experiment1.md).
- `experiments/boundary_augmented_network.py` — PyTorch prototype comparing boundary and random weight perturbations. See [boundary_augmented_network](experiments/boundary_augmented_network.md).
- `experiments/boundary_observer_study.py` — Controlled forecasting and copied-weight interventions using boundary observations. See [boundary_observer_study](experiments/boundary_observer_study.md).
- `experiments/clayworth_simplicial_regression.py` — Finite regression checks for proposed simplicial-object constructions. See [clayworth_simplicial_regression](experiments/clayworth_simplicial_regression.md).
- `experiments/consolidated_experiment.py` — Random-matrix comparisons of boundaries, degeneracy, and distribution summaries. See [consolidated_experiment](experiments/consolidated_experiment.md).
- `experiments/consolidated_experiment_with_n01.py` — The random-matrix study with an additional Gaussian input family. See [consolidated_experiment_with_n01](experiments/consolidated_experiment_with_n01.md).
- `experiments/convex_boundary_study.py` — Validation-tuned matrix regression with boundary and matched penalty controls. See [convex_boundary_study](experiments/convex_boundary_study.md).
- `experiments/cospectral_proof.py` — Exhaustive mod-two boundary ranks for an atlas pair and a zero-boundary test for Petersen labelings. See [cospectral_proof](experiments/cospectral_proof.md).
- `experiments/count_standard_basis_tensors.py` — Enumeration of coordinate tensors accepted by the degeneracy test. See [count_standard_basis_tensors](experiments/count_standard_basis_tensors.md).
- `experiments/degenerate_counterexample.py` — A matrix example separating selected filler recovery from filler uniqueness. See [degenerate_counterexample](experiments/degenerate_counterexample.md).
- `experiments/discrepancy_test.py` — Sampled reconstruction comparisons with tabular and plot outputs. See [discrepancy_test](experiments/discrepancy_test.md).
- `experiments/examples_range_tensors.py` — Fixed range-tensor horn, degeneracy, and filler comparisons. See [examples_range_tensors](experiments/examples_range_tensors.md).
- `experiments/filler_combinatorics.py` — Symbolic comparison of uncovered horn coordinates and Moore filler corrections. See [filler_combinatorics](experiments/filler_combinatorics.md).
- `experiments/fixed_index_face_independence.py` — Floating-point rank checks on surviving coordinate images of each face. See [fixed_index_face_independence](experiments/fixed_index_face_independence.md).
- `experiments/g115_g117_full_enumeration.py` — Exact mod-two boundary-rank histograms for all 720 labelings of each graph. See [g115_g117_full_enumeration](experiments/g115_g117_full_enumeration.md).
- `experiments/homology_basis.py` — Exact normalized homology of tensor-generated submodules and sparse graph representations. See [homology_basis](experiments/homology_basis.md).
- `experiments/homotopy_constraint_verification.py` — Comparison of horn-support tuples with the common kernel of included faces. See [homotopy_constraint_verification](experiments/homotopy_constraint_verification.md).
- `experiments/homotopy_constraint_verification_with_independence_test.py` — The horn-support diagnostic with separate face-image independence tests. See [homotopy_constraint_verification_with_independence_test](experiments/homotopy_constraint_verification_with_independence_test.md).
- `experiments/horn_map_reduce.py` — Dask enumeration of coordinates absent from the included horn faces. See [horn_map_reduce](experiments/horn_map_reduce.md).
- `experiments/loss_experiment.py` — Reconstruction-loss distributions over 10,000 random-matrix trials. See [loss_experiment](experiments/loss_experiment.md).
- `experiments/matrix_boundary_homotopy_identity.py` — Signed and mod-two boundary checks under induced vertex permutations. See [matrix_boundary_homotopy_identity](experiments/matrix_boundary_homotopy_identity.md).
- `experiments/n_cycle_conjugation.py` — Cyclic relabeling and signed or mod-two boundary comparisons. See [n_cycle_conjugation](experiments/n_cycle_conjugation.md).
- `experiments/n_hypergroupoid_conjecture.py` — Checked Moore fillers and the full-module horn-kernel criterion on seeded tensor samples. See [n_hypergroupoid_conjecture](experiments/n_hypergroupoid_conjecture.md).
- `experiments/neural_boundary_study.py` — Paired-seed digits models with validation tuning and neural symmetry controls. See [neural_boundary_study](experiments/neural_boundary_study.md).
- `experiments/normalized_moore_complex_homology.py` — Exact homology of normalized quotients by the included horn faces. See [normalized_moore_complex_homology](experiments/normalized_moore_complex_homology.md).
- `experiments/notebooks/Homology_calculation.ipynb` — Symbolic matrix boundary, preimage equations, and kernel comparisons. See [the notebook guide](experiments/tensor_boundary_notebooks.md).
- `experiments/notebooks/cocalc_compatible_tensor_ops.ipynb` — Shared symbolic horn, rank, and degeneracy examples. See [the notebook guide](api/sagemath_compatible_tensor_ops.md).
- `experiments/notebooks/homology_syzygy.ipynb` — Symbolic equations for a matrix preimage. See [the notebook guide](experiments/tensor_boundary_notebooks.md).
- `experiments/notebooks/homology_syzygy_fixed.ipynb` — Polynomial-ring setup and symbolic preimage equations. See [the notebook guide](experiments/tensor_boundary_notebooks.md).
- `experiments/notebooks/section_6_2_eta_delta_boundary_check.ipynb` — Integer column-space membership for a word-deletion boundary. See [the notebook guide](experiments/tensor_boundary_notebooks.md).
- `experiments/notebooks/petersen_coset_enumeration.ipynb` — SageMath notebook and script for Petersen permutation cosets. See [petersen_coset_enumeration](experiments/petersen_coset_enumeration.md).
- `experiments/notebooks/petersen_coset_enumeration.sage` — SageMath notebook and script for Petersen permutation cosets. See [petersen_coset_enumeration](experiments/petersen_coset_enumeration.md).
- `experiments/permutation_statistics.py` — Command-line cycle statistics and plots with bounded power tests. See [permutation_statistics](experiments/permutation_statistics.md).
- `experiments/petersen_brd_right_cosets.py` — Exact Petersen deletion-boundary ranks from 30,240 distinct adjacency matrices. See [petersen_brd_right_cosets](experiments/petersen_brd_right_cosets.md).
- `experiments/random_zero_one_mask_cocycle_test.py` — Seeded checks of boundary linearity and cycles after masking and relabeling. See [random_zero_one_mask_cocycle_test](experiments/random_zero_one_mask_cocycle_test.md).
- `experiments/sagemath_compatible_tensor_ops.py` — Demonstration of the package's SageMath or SymPy symbolic backend. See [sagemath_compatible_tensor_ops](api/sagemath_compatible_tensor_ops.md).
- `experiments/sagemath_normalized_tensor_homology.py` — Exact tensor and relative horn homology over rational or prime fields using SymPy. See [sagemath_normalized_tensor_homology](experiments/sagemath_normalized_tensor_homology.md).
- `experiments/shape_generator.py` — Bounded shape enumeration ordered by degree, axis count, and offset sum. See [shape_generator](experiments/shape_generator.md).
- `experiments/simplicial_permutation_group.py` — Finite checks of cycle deletion under vertex removal. See [simplicial_permutation_group](experiments/simplicial_permutation_group.md).
- `experiments/verify_degenerate_preference.py` — Finite checks of the algorithm's degenerate filler selection. See [verify_degenerate_preference](experiments/verify_degenerate_preference.md).
- `experiments/verify_injective_face_map.py` — Injectivity checks on coordinate tensors surviving a fixed face. See [verify_injective_face_map](experiments/verify_injective_face_map.md).

## Stored studies

- `experiments/results/boundary_observer_study/final_metrics.csv` — Per-trajectory final training and held-out task metrics. See [boundary_observer_study](experiments/boundary_observer_study.md).
- `experiments/results/boundary_observer_study/forecast_metrics.csv` — Held-out forecasting results after observer tuning. See [boundary_observer_study](experiments/boundary_observer_study.md).
- `experiments/results/boundary_observer_study/shadow_interventions.csv` — Training and validation effects of matched copied-weight interventions. See [boundary_observer_study](experiments/boundary_observer_study.md).
- `experiments/results/boundary_observer_study/summary.json` — Configuration, diagnostics, aggregate results, and software versions. See [boundary_observer_study](experiments/boundary_observer_study.md).
- `experiments/results/boundary_observer_study/trajectories.csv` — Measurements at the recorded training checkpoints. See [boundary_observer_study](experiments/boundary_observer_study.md).
- `experiments/results/convex_boundary_study/signal1.csv` — Per-run results for the indicated signal level. See [convex_boundary_study](experiments/convex_boundary_study.md).
- `experiments/results/convex_boundary_study/signal1.json` — Protocol, diagnostics, and summaries for the indicated signal level. See [convex_boundary_study](experiments/convex_boundary_study.md).
- `experiments/results/convex_boundary_study/signal2.csv` — Per-run results for the indicated signal level. See [convex_boundary_study](experiments/convex_boundary_study.md).
- `experiments/results/convex_boundary_study/signal2.json` — Protocol, diagnostics, and summaries for the indicated signal level. See [convex_boundary_study](experiments/convex_boundary_study.md).
- `experiments/results/neural_boundary_study/confirmation_runs.csv` — Paired confirmation runs after validation tuning. See [neural_boundary_study](experiments/neural_boundary_study.md).
- `experiments/results/neural_boundary_study/summary.json` — Selected methods, result summaries, and invariance diagnostics. See [neural_boundary_study](experiments/neural_boundary_study.md).
- `experiments/results/neural_boundary_study/tuning_runs.csv` — Validation-search results for the evaluated methods. See [neural_boundary_study](experiments/neural_boundary_study.md).

## Other artifacts

- `experiments/boundary_observer_report.md` — Protocol, stored results, validity checks, and interpretation of the observer study.
- `experiments/results/loss_experiment/experiment_results.csv` — Shape, reconstruction-loss, and test statistics using one fixed comparison array per shape.
- `experiments/results/sagemath_compatible_tensor_ops/output.txt` — Stored symbolic horn and filler comparisons beginning with shape (3,3).
- `experiments/results/training_accuracy/figures/Training and Validation Accuracy Figure_1.png` — Training and validation figure; no generating command or configuration is recorded with the file.
- `experiments/results/training_accuracy/figures/Training and Validation Accuracy Figure_2.png` — Training and validation figure; no generating command or configuration is recorded with the file.
- `experiments/results/permutation_statistics/figures/diagonal_vs_properties.png` — Summary figure. See [permutation_statistics](experiments/permutation_statistics.md).
- `experiments/results/discrepancy_test/figures/discrepancy_analysis.png` — Summary figure. See [discrepancy_test](experiments/discrepancy_test.md).
- `experiments/results/permutation_statistics/figures/properties_by_size.png` — Summary figure. See [permutation_statistics](experiments/permutation_statistics.md).
- `experiments/results/permutation_statistics/figures/properties_vs_similarity.png` — Summary figure. See [permutation_statistics](experiments/permutation_statistics.md).
- `experiments/results/permutation_statistics/figures/similarity_distribution.png` — Summary figure. See [permutation_statistics](experiments/permutation_statistics.md).

## Diagnostic logs

- `experiments/results/finite_difference_analysis/logs/analysis_log_2025-06-24_21-18-47.txt` — Finite-difference analysis output dated 2025-06-24. See [the analysis helpers](experiments/finite_difference_analysis.md).
- `experiments/results/finite_difference_analysis/logs/analysis_log_2025-06-24_21-34-07.txt` — Finite-difference analysis output dated 2025-06-24. See [the analysis helpers](experiments/finite_difference_analysis.md).
- `experiments/results/finite_difference_analysis/logs/analysis_log_2025-06-24_21-41-12.txt` — Finite-difference analysis output dated 2025-06-24. See [the analysis helpers](experiments/finite_difference_analysis.md).
- `experiments/results/finite_difference_analysis/logs/analysis_log_2025-06-24_21-55-05.txt` — Finite-difference analysis output dated 2025-06-24. See [the analysis helpers](experiments/finite_difference_analysis.md).
- `experiments/results/finite_difference_analysis/logs/analysis_log_2025-06-24_22-06-50.txt` — Finite-difference analysis output dated 2025-06-24. See [the analysis helpers](experiments/finite_difference_analysis.md).
- `experiments/results/finite_difference_analysis/logs/analysis_log_2025-06-24_23-02-26.txt` — Finite-difference analysis output dated 2025-06-24. See [the analysis helpers](experiments/finite_difference_analysis.md).
- `experiments/results/finite_difference_analysis/logs/analysis_log_2025-06-24_23-04-28.txt` — Finite-difference analysis output dated 2025-06-24. See [the analysis helpers](experiments/finite_difference_analysis.md).
- `experiments/results/finite_difference_analysis/logs/analysis_log_2025-06-28_13-24-12.txt` — Finite-difference analysis output dated 2025-06-28. See [the analysis helpers](experiments/finite_difference_analysis.md).
- `experiments/results/finite_difference_analysis/logs/analysis_log_2025-06-28_13-45-56.txt` — Finite-difference analysis output dated 2025-06-28. See [the analysis helpers](experiments/finite_difference_analysis.md).
- `experiments/results/finite_difference_analysis/logs/analysis_log_2025-06-28_13-47-00.txt` — Finite-difference analysis output dated 2025-06-28. See [the analysis helpers](experiments/finite_difference_analysis.md).
- `experiments/results/finite_difference_analysis/logs/analysis_log_2025-06-28_13-48-00.txt` — Finite-difference analysis output dated 2025-06-28. See [the analysis helpers](experiments/finite_difference_analysis.md).
- `experiments/results/finite_difference_analysis/logs/analysis_log_2025-06-28_19-36-46.txt` — Finite-difference analysis output dated 2025-06-28. See [the analysis helpers](experiments/finite_difference_analysis.md).
- `experiments/results/verify_injective_face_map/logs/injective_out.txt` — Coordinate-face injectivity output. See [the face-map guide](experiments/verify_injective_face_map.md).
