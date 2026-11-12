"""Compare Gaussian boundary and independent-array reconstruction losses."""

import argparse
import ast
import csv
import logging
import os
from typing import Tuple

import numpy as np
from scipy import stats
from simplicial_tensors.tensor_ops import bdry, degen


def reconstruct_T_from_S(S: np.ndarray, T_shape: Tuple[int, ...]) -> np.ndarray:
    """
    Reconstruct an approximation of T from S by degenerating (duplicating) elements along each axis.

    Parameters:
        S (np.ndarray): Tensor of shape (a_1 - 1, a_2 - 1, ..., a_k - 1)
        T_shape (tuple): Shape of T (a_1, a_2, ..., a_k)

    Returns:
        np.ndarray: Reconstructed tensor of shape T_shape
    """
    # Apply degen with k=0 to duplicate the 0-th hypercolumn along each axis
    T_hat = degen(S, 0)
    
    # After degeneracy, ensure that T_hat matches T_shape
    if T_hat.shape != T_shape:
        raise ValueError(f"Reconstructed tensor shape {T_hat.shape} does not match target shape {T_shape}.")
    
    return T_hat

def compute_loss(T: np.ndarray, S: np.ndarray) -> float:
    """
    Compute the loss between T and the reconstructed T from S.

    Parameters:
        T (np.ndarray): Original tensor of shape T_shape
        S (np.ndarray): Tensor of shape (a_1 - 1, a_2 - 1, ..., a_k - 1)

    Returns:
        float: The norm of the difference between T and reconstructed T_hat
    """
    T_hat = reconstruct_T_from_S(S, T.shape)
    T_norm = T / np.linalg.norm(T) if np.linalg.norm(T) != 0 else T
    T_hat_norm = T_hat / np.linalg.norm(T_hat) if np.linalg.norm(T_hat) != 0 else T_hat
    loss = np.linalg.norm(T_norm - T_hat_norm)
    return loss

# ----------------------------
# Experiment Execution
# ----------------------------

def run_experiment(
    tensor_shape: Tuple[int, ...],
    num_trials: int = 10000,
    mean: float = 0.0,
    std: float = 1.0,
    *, seed: int = 12345, progress: bool = False
) -> dict:
    """
    Run the experiment for a specific tensor shape.

    Parameters:
        tensor_shape (Tuple[int, ...]): Shape of tensor T.
        num_trials (int): Number of trials to run.
        mean (float): Mean of the normal distribution for tensor generation.
        std (float): Standard deviation of the normal distribution for tensor generation.

    Returns:
        dict: A dictionary containing experiment results.
    """
    if num_trials < 2 or not tensor_shape or min(tensor_shape) < 2:
        raise ValueError("Use at least two trials and positive boundary dimensions")
    if std < 0:
        raise ValueError("std must be nonnegative")
    rng = np.random.default_rng(seed)
    trials = range(num_trials)
    if progress:
        from tqdm import tqdm
        trials = tqdm(trials, desc=f"Tensor Shape {tensor_shape}", unit="trial")
    logging.info(f"Starting experiments for tensor shape: {tensor_shape}, seed={seed}")
    losses_S = []
    losses_boundary = []
    for _ in trials:
        T = rng.normal(mean, std, size=tensor_shape)
        partial_T = bdry(T)
        S = rng.normal(mean, std, size=partial_T.shape)
        
        # Compute losses
        loss_S = compute_loss(T, S)
        loss_boundary = compute_loss(T, partial_T)
        
        losses_S.append(loss_S)
        losses_boundary.append(loss_boundary)

    # Convert to numpy arrays for efficient computation
    losses_S = np.array(losses_S)
    losses_boundary = np.array(losses_boundary)

    # Calculate statistical metrics
    avg_loss_S = np.mean(losses_S)
    avg_loss_boundary = np.mean(losses_boundary)
    inequality_holds = np.sum(losses_S >= losses_boundary)
    inequality_percentage = (inequality_holds / num_trials) * 100
    std_loss_S = np.std(losses_S, ddof=1)
    std_loss_boundary = np.std(losses_boundary, ddof=1)
    t_statistic, p_value = stats.ttest_rel(losses_S, losses_boundary)
    mean_diff = avg_loss_S - avg_loss_boundary
    pooled_std = np.sqrt((std_loss_S ** 2 + std_loss_boundary ** 2) / 2)
    cohen_d = mean_diff / pooled_std if pooled_std != 0 else (0.0 if mean_diff == 0 else np.copysign(np.inf, mean_diff))

    # Determine per-axis parity
    axis_parities = ['Even' if dim % 2 == 0 else 'Odd' for dim in tensor_shape]
    num_even_axes = axis_parities.count('Even')
    num_odd_axes = axis_parities.count('Odd')

    # Compile results into a dictionary
    results_dict = {
        'Tensor Shape': tensor_shape,
        'Seed': seed,
        'Trials': num_trials,
        'Number of Axes': len(tensor_shape),
        'Axis Parities': axis_parities,
        'Number of Even Axes': num_even_axes,
        'Number of Odd Axes': num_odd_axes,
        'Average Loss T & S': avg_loss_S,
        'Average Loss T & ∂T': avg_loss_boundary,
        'Inequality Holds (%)': inequality_percentage,
        'Standard Deviation Loss S': std_loss_S,
        'Standard Deviation Loss ∂T': std_loss_boundary,
        'Paired t-test Statistic': t_statistic,
        'p-value': p_value,
        'Cohen\'s d': cohen_d
    }

    logging.info(f"Completed experiments for tensor shape: {tensor_shape}")
    logging.info(f"  Number of Even Axes: {num_even_axes}")
    logging.info(f"  Number of Odd Axes: {num_odd_axes}")
    logging.info(f"  Average Loss between T and S: {avg_loss_S:.6f}")
    logging.info(f"  Average Loss between T and ∂T: {avg_loss_boundary:.6f}")
    logging.info(f"  Inequality Loss(T, S) ≥ Loss(T, ∂T) holds in {inequality_percentage:.2f}% of trials")
    logging.info(f"  Standard Deviation of Loss(T, S): {std_loss_S:.6f}")
    logging.info(f"  Standard Deviation of Loss(∂T): {std_loss_boundary:.6f}")
    logging.info(f"  Paired t-test Statistic: {t_statistic:.6f}")
    logging.info(f"  p-value: {p_value}")
    logging.info(f"  Cohen's d (Effect Size): {cohen_d:.6f}")
    logging.info("-" * 60)

    return results_dict

# ----------------------------
# Plotting Functionality
# ----------------------------

def plot_results(csv_file: str, *, show: bool = False):
    """
    Generate and save plots based on the experiment results stored in a CSV file.

    Parameters:
        csv_file (str): Path to the CSV file containing experiment results.
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Check if the CSV file exists
    if not os.path.exists(csv_file):
        logging.error(f"CSV file '{csv_file}' not found. Skipping plotting.")
        return

    # Load the results
    df = pd.read_csv(csv_file)

    # Ensure that 'Tensor Shape' is treated as a categorical variable
    df['Tensor Shape'] = df['Tensor Shape'].astype(str)

    # Parse axis parities from the 'Axis Parities' column safely
    df['Axis Parity List'] = df['Axis Parities'].apply(lambda x: ast.literal_eval(x))

    # Calculate the number of even and odd axes
    df['Number of Even Axes'] = df['Axis Parity List'].apply(lambda x: x.count('Even'))
    df['Number of Odd Axes'] = df['Axis Parity List'].apply(lambda x: x.count('Odd'))

    # Set plot style
    sns.set(style="whitegrid")

    # Create 'plots' directory if it doesn't exist
    plots_dir = os.path.join(os.getcwd(), "plots")
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
        logging.info(f"Created 'plots' directory at {plots_dir}")

    # ----------------------------
    # Plot 1: Average Loss Comparison
    # ----------------------------
    plt.figure(figsize=(14, 7))
    df_melted = df.melt(id_vars=["Tensor Shape"], value_vars=["Average Loss T & S", "Average Loss T & ∂T"],
                        var_name="Condition", value_name="Average Loss")
    sns.barplot(x="Tensor Shape", y="Average Loss", hue="Condition", data=df_melted, palette=["skyblue", "lightgreen"])
    plt.title("Average Loss Comparison Across Tensor Shapes")
    plt.xlabel("Tensor Shape")
    plt.ylabel("Average Loss")
    plt.xticks(rotation=45)
    plt.legend(title="Condition")
    plt.tight_layout()
    plot_path = os.path.join(plots_dir, "average_loss_comparison.png")
    plt.savefig(plot_path)
    if show:
        plt.show(block=True)
    plt.close()
    logging.info(f"Plot 'average_loss_comparison.png' has been saved to {plots_dir}.")

    # ----------------------------
    # Plot 2: Inequality Holding Percentage vs. Number of Even Axes
    # ----------------------------
    plt.figure(figsize=(14, 7))
    sns.scatterplot(x="Number of Even Axes", y="Inequality Holds (%)", hue="Number of Odd Axes",
                    palette="viridis", size="Number of Odd Axes", sizes=(50, 200), data=df, alpha=0.7)
    plt.title("Inequality Holds (%) vs. Number of Even Axes")
    plt.xlabel("Number of Even Axes")
    plt.ylabel("Inequality Holds (%)")
    plt.legend(title="Number of Odd Axes", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plot_path = os.path.join(plots_dir, "inequality_holds_vs_even_axes.png")
    plt.savefig(plot_path)
    if show:
        plt.show(block=True)
    plt.close()
    logging.info(f"Plot 'inequality_holds_vs_even_axes.png' has been saved to {plots_dir}.")

    # ----------------------------
    # Plot 3: Paired t-test Statistic and Cohen's d vs. Number of Even Axes
    # ----------------------------
    plt.figure(figsize=(14, 7))
    df_plot = df[['Tensor Shape', 'Paired t-test Statistic', 'Cohen\'s d', 'Number of Even Axes']]
    # Melt for easier plotting
    df_plot_melted = df_plot.melt(id_vars=['Tensor Shape', 'Number of Even Axes'], value_vars=['Paired t-test Statistic', 'Cohen\'s d'],
                                  var_name='Metric', value_name='Value')
    sns.barplot(x='Number of Even Axes', y='Value', hue='Metric', data=df_plot_melted, palette=["salmon", "lightblue"])
    plt.title("Paired t-test Statistic and Cohen's d vs. Number of Even Axes")
    plt.xlabel("Number of Even Axes")
    plt.ylabel("Metric Value")
    plt.xticks(rotation=45)
    plt.legend(title="Metric", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plot_path = os.path.join(plots_dir, "statistical_metrics_vs_even_axes.png")
    plt.savefig(plot_path)
    if show:
        plt.show(block=True)
    plt.close()
    logging.info(f"Plot 'statistical_metrics_vs_even_axes.png' has been saved to {plots_dir}.")

    # ----------------------------
    # Plot 4: Cohen's d (Effect Size) vs. Number of Even Axes
    # ----------------------------
    plt.figure(figsize=(14, 7))
    sns.scatterplot(x="Number of Even Axes", y="Cohen's d", hue="Number of Odd Axes",
                    palette="Set2", size="Number of Odd Axes", sizes=(50, 200), data=df, alpha=0.7)
    plt.title("Cohen's d (Effect Size) vs. Number of Even Axes")
    plt.xlabel("Number of Even Axes")
    plt.ylabel("Cohen's d")
    plt.legend(title="Number of Odd Axes", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plot_path = os.path.join(plots_dir, "cohens_d_vs_even_axes.png")
    plt.savefig(plot_path)
    if show:
        plt.show(block=True)
    plt.close()
    logging.info(f"Plot 'cohens_d_vs_even_axes.png' has been saved to {plots_dir}.")

    # ----------------------------
    # Plot 5: Distribution of Axis Parity
    # ----------------------------
    plt.figure(figsize=(10, 6))
    # Create a single list of all axis parities
    all_parities = [parity for sublist in df['Axis Parity List'] for parity in sublist]
    sns.countplot(x=all_parities)  # Removed 'palette' to fix warnings
    plt.title("Distribution of Axis Parity Across All Tensor Axes")
    plt.xlabel("Axis Parity")
    plt.ylabel("Count")
    plt.tight_layout()
    plot_path = os.path.join(plots_dir, "axis_parity_distribution.png")
    plt.savefig(plot_path)
    if show:
        plt.show(block=True)
    plt.close()
    logging.info(f"Plot 'axis_parity_distribution.png' has been saved to {plots_dir}.")

    # ----------------------------
    # Plot 6: Average Loss vs. Number of Even Axes
    # ----------------------------
    plt.figure(figsize=(14, 7))
    sns.boxplot(x="Number of Even Axes", y="Average Loss T & S", data=df, showfliers=False, color='skyblue')
    sns.boxplot(x="Number of Even Axes", y="Average Loss T & ∂T", data=df, showfliers=False, color='lightgreen')
    plt.title("Average Loss vs. Number of Even Axes")
    plt.xlabel("Number of Even Axes")
    plt.ylabel("Average Loss")
    plt.legend(labels=["Average Loss T & S", "Average Loss T & ∂T"], bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plot_path = os.path.join(plots_dir, "average_loss_vs_even_axes.png")
    plt.savefig(plot_path)
    if show:
        plt.show(block=True)
    plt.close()
    logging.info(f"Plot 'average_loss_vs_even_axes.png' has been saved to {plots_dir}.")

    # ----------------------------
    # Plot 7: Heat Map of Effect Sizes (Cohen's d)
    # ----------------------------
    plt.figure(figsize=(10, 8))
    # Pivot the data to create a matrix suitable for heatmap
    # Rows: Number of Even Axes
    # Columns: Number of Odd Axes
    # Values: Average Cohen's d
    heatmap_data = df.groupby(['Number of Even Axes', 'Number of Odd Axes'])['Cohen\'s d'].mean().reset_index()
    heatmap_pivot = heatmap_data.pivot(index='Number of Even Axes', columns='Number of Odd Axes', values='Cohen\'s d')
    
    # Create the heatmap
    sns.heatmap(heatmap_pivot, annot=True, fmt=".2f", cmap="YlGnBu", cbar_kws={'label': "Cohen's d"})
    plt.title("Heat Map of Effect Sizes (Cohen's d)")
    plt.xlabel("Number of Odd Axes")
    plt.ylabel("Number of Even Axes")
    plt.tight_layout()
    plot_path = os.path.join(plots_dir, "effect_size_heatmap.png")
    plt.savefig(plot_path)
    if show:
        plt.show(block=True)
    plt.close()
    logging.info(f"Plot 'effect_size_heatmap.png' has been saved to {plots_dir}.")

    logging.info("All plots have been generated, saved to the 'plots' directory.")

# ----------------------------
# Main Execution Function
# ----------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trials', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--mean', type=float, default=0.0)
    parser.add_argument('--std', type=float, default=1.0)
    parser.add_argument('--shape', action='append', help='Comma-separated axis lengths')
    parser.add_argument('--output', default='experiment_results.csv')
    parser.add_argument('--log', default='experiment.log')
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--show', action='store_true', help='Display plot windows as well as saving them')
    parser.add_argument('--progress', action='store_true')
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(args.log, mode='w', encoding='utf-8'),
                                  logging.StreamHandler()])
    shapes = [(5, 5, 5), (4, 5, 5), (5, 6, 7), (4, 6, 8),
              (6, 7, 8, 9), (8, 9, 10), (5, 7, 9, 10), (5, 7, 9, 11),
              (6, 6, 6, 6), (7, 7, 7, 7), (9, 9, 9), (9, 9, 11, 11)]
    if args.shape:
        shapes = [tuple(int(n) for n in shape.split(',')) for shape in args.shape]
    results = [run_experiment(shape, args.trials, args.mean, args.std,
                              seed=args.seed, progress=args.progress) for shape in shapes]
    with open(args.output, 'w', newline='', encoding='utf-8') as output:
        writer = csv.DictWriter(output, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    if args.plot or args.show:
        plot_results(args.output, show=args.show)
    return results


if __name__ == '__main__':
    main()
