import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_loss_file(file_path):
    """Read loss values from a txt file"""
    with open(file_path, "r") as f:
        lines = f.readlines()
    # Convert to float, removing empty lines
    losses = [float(line.strip()) for line in lines if line.strip()]
    return np.array(losses)


def calculate_statistics(losses):
    """Calculate mean, median, and std for a list of losses"""
    return {"mean": np.mean(losses), "median": np.median(losses), "std": np.std(losses)}


def analyze_mpe_losses():
    """Analyze specific mpe loss files and plot statistics"""

    # Path to mpe_losses folder
    mpe_losses_path = Path("diffusion_model/mpe_losses")

    # Specify the iterations we want to analyze
    target_iterations = [1, 2, 6, 10, 14, 18, 22, 26, 30, 34, 38]

    # Get specific txt files
    loss_files = []
    for iteration in target_iterations:
        file_path = mpe_losses_path / f"mpe_loss_{iteration}.txt"
        if file_path.exists():
            loss_files.append(str(file_path))

    # Extract iteration numbers and calculate statistics
    iterations = []
    means = []
    medians = []
    stds = []

    for file_path in loss_files:
        # Extract iteration number from filename
        filename = os.path.basename(file_path)
        iteration = int(filename.replace("mpe_loss_", "").replace(".txt", ""))

        # Read losses and calculate statistics
        losses = read_loss_file(file_path)
        stats = calculate_statistics(losses)

        iterations.append(iteration)
        means.append(stats["mean"])
        medians.append(stats["median"])
        stds.append(stats["std"])

        # Calculate epoch for this iteration
        samples_per_step = 16 * 2  # train_batch_size * gradient_accumulate_every
        epoch = (iteration * samples_per_step) / 11000 * 1000  # dataset_size * 1000

        print(
            f"Iteration {iteration} (Epoch {epoch:.0f}): Mean={stats['mean']:.2f}, Median={stats['median']:.2f}, Std={stats['std']:.2f}"
        )

    # Sort by iteration number
    sorted_data = sorted(zip(iterations, means, medians, stds))
    iterations, means, medians, stds = zip(*sorted_data)

    # Calculate epochs for x-axis
    train_batch_size = 16
    gradient_accumulate_every = 2
    dataset_size = 11000

    samples_per_step = train_batch_size * gradient_accumulate_every
    epochs = [
        (step_index * samples_per_step) / dataset_size * 1000
        for step_index in iterations
    ]

    # Create the plot
    plt.figure(figsize=(12, 8))

    # Plot mean, median, and std data points with custom colors
    plt.plot(
        epochs,
        means,
        "o-",
        label="Mean",
        linewidth=4,
        markersize=8,
        color="#FF6B6B",
        markerfacecolor="#FF6B6B",
        markeredgecolor="#CC5555",
        markeredgewidth=1.5,
    )
    plt.plot(
        epochs,
        medians,
        "s-",
        label="Median",
        linewidth=4,
        markersize=8,
        color="#2ECC71",
        markerfacecolor="#2ECC71",
        markeredgecolor="#27AE60",
        markeredgewidth=1.5,
    )
    plt.plot(
        epochs,
        stds,
        "^-",
        label="Standard Deviation",
        linewidth=4,
        markersize=8,
        color="#9B59B6",
        markerfacecolor="#9B59B6",
        markeredgecolor="#8E44AD",
        markeredgewidth=1.5,
    )

    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss Value, %", fontsize=12)
    plt.title("MPE Loss Statistics Over Epochs", fontsize=14, fontweight="bold")
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    # Improve layout
    plt.tight_layout()

    # Save the plot
    output_path = Path("diffusion_model/mpe_loss_analysis.pdf")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nPlot saved to: {output_path}")

    # Show the plot
    plt.show()

    return iterations, means, medians, stds


if __name__ == "__main__":
    # Run the analysis
    iterations, means, medians, stds = analyze_mpe_losses()

    # Print summary statistics
    print("\nSummary:")
    print(f"Total iterations analyzed: {len(iterations)}")
    print(f"Mean loss range: {min(means):.2f} - {max(means):.2f}")
    print(f"Median loss range: {min(medians):.2f} - {max(medians):.2f}")
    print(f"Std range: {min(stds):.2f} - {max(stds):.2f}")
