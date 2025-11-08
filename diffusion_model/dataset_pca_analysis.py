from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def load_and_analyze_dataset(csv_path):
    """Load dataset and perform PCA analysis"""

    print(f"Loading dataset from: {csv_path}")

    # Load the dataset
    df = pd.read_csv(csv_path)
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    # Display first few rows
    print("\nFirst 5 rows:")
    print(df.head())

    return df


def create_pca_visualization(df):
    """Create PCA visualization of the dataset"""

    # Separate features for different analyses
    # Assuming the dataset has columns for coordinates, radii, and angles
    # We'll need to identify which columns correspond to what

    # Let's first check what columns we have
    print(f"\nDataset columns: {list(df.columns)}")

    # For now, let's assume the structure and create a flexible analysis
    # You may need to adjust column names based on your actual dataset

    # Try to identify coordinate columns (assuming they might be named x, y, z or similar)
    coord_cols = []
    radius_cols = []
    angle_cols = []

    for col in df.columns:
        col_lower = col.lower()
        if "coord_" in col_lower:
            coord_cols.append(col)
        elif "radius_" in col_lower:
            radius_cols.append(col)
        elif "dscs_angle_" in col_lower:
            angle_cols.append(col)

    print(f"Identified coordinate columns: {coord_cols}")
    print(f"Identified radius columns: {radius_cols}")
    print(f"Identified angle columns: {angle_cols}")

    # If we can't identify specific columns, use all numeric columns
    if not coord_cols and not radius_cols:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        print(f"Using all numeric columns: {numeric_cols}")

        # For PCA, we'll use all numeric features
        features = df[numeric_cols].values

        # Standardize the features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        # Perform PCA
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(features_scaled)

        # Create the visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        # Plot 1: PCA coordinates and radii (if available)
        if radius_cols:
            # Use the first radius column for point sizes
            radii = df[radius_cols[0]].values
            scatter = ax1.scatter(
                pca_result[:, 0],
                pca_result[:, 1],
                c=radii,
                cmap="viridis",
                alpha=0.6,
                s=50,
            )
            ax1.set_xlabel("First Principal Component")
            ax1.set_ylabel("Second Principal Component")
            ax1.set_title("PCA: Coordinates and Radii")
            plt.colorbar(scatter, ax=ax1, label="Radius")
        else:
            # Just plot PCA coordinates
            ax1.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.6, s=50)
            ax1.set_xlabel("First Principal Component")
            ax1.set_ylabel("Second Principal Component")
            ax1.set_title("PCA: Dataset Coordinates")

        ax1.grid(True, alpha=0.3)

        # Plot 2: Angle distributions
        if angle_cols:
            # Create subplots for each angle
            num_angles = len(angle_cols)
            if num_angles <= 4:
                fig2, axes = plt.subplots(2, 2, figsize=(12, 10))
                axes = axes.flatten()
            else:
                cols = int(np.ceil(np.sqrt(num_angles)))
                rows = int(np.ceil(num_angles / cols))
                fig2, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
                if num_angles == 1:
                    axes = [axes]
                else:
                    axes = axes.flatten()

            for i, angle_col in enumerate(angle_cols):
                if i < len(axes):
                    axes[i].hist(
                        df[angle_col].values, bins=30, alpha=0.7, edgecolor="black"
                    )
                    axes[i].set_xlabel("DSCS Value")
                    axes[i].set_ylabel("Number of Samples")
                    axes[i].set_title(f"DSCS Distribution on {angle_col}$\degree$")
                    axes[i].grid(True, alpha=0.3)

            # Hide empty subplots
            for i in range(num_angles, len(axes)):
                axes[i].set_visible(False)

            plt.tight_layout()
            plt.savefig(
                "diffusion_model/angle_distributions.png", dpi=300, bbox_inches="tight"
            )
            plt.show()
        else:
            # If no angle columns found, show feature distributions
            num_features = min(4, len(numeric_cols))
            fig2, axes = plt.subplots(2, 2, figsize=(12, 10))
            axes = axes.flatten()

            for i in range(num_features):
                axes[i].hist(
                    df[numeric_cols[i]].values, bins=30, alpha=0.7, edgecolor="black"
                )
                axes[i].set_xlabel("Value")
                axes[i].set_ylabel("Frequency")
                axes[i].set_title(f"Distribution of {numeric_cols[i]}")
                axes[i].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(
                "diffusion_model/feature_distributions.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.show()

        # Save the main PCA plot
        plt.tight_layout()
        plt.savefig(
            "diffusion_model/dataset_pca_analysis.png", dpi=300, bbox_inches="tight"
        )
        plt.show()

        # Print PCA information
        print(f"\nPCA Explained Variance Ratio: {pca.explained_variance_ratio_}")
        print(f"Total Explained Variance: {sum(pca.explained_variance_ratio_):.3f}")

        return pca_result, pca

    else:
        # If we have identified specific column types, create more targeted analysis
        print("Creating targeted analysis based on identified column types...")

        # Use DSCS angles for PCA
        features_for_pca = angle_cols
        print(f"Using features for PCA: {features_for_pca}")

        # Prepare features for PCA
        features = df[features_for_pca].values

        # Standardize the features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        # Perform PCA
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(features_scaled)

        # Create the visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        # Plot 1: PCA of DSCS angles with radius coloring
        # Use average radius for coloring
        avg_radii = df[radius_cols].mean(axis=1).values
        scatter = ax1.scatter(
            pca_result[:, 0],
            pca_result[:, 1],
            c=avg_radii,
            cmap="viridis",
            alpha=0.6,
            s=50,
        )
        ax1.set_xlabel("First Principal Component")
        ax1.set_ylabel("Second Principal Component")
        ax1.set_title("PCA: DSCS")
        plt.colorbar(scatter, ax=ax1, label="Average Radius")
        ax1.grid(True, alpha=0.3)

        # Plot 2: Empty for now, will be filled with angle info
        ax2.text(
            0.5,
            0.5,
            "PCA Analysis Complete\nCheck angle_distributions.png\nfor angle distributions",
            ha="center",
            va="center",
            transform=ax2.transAxes,
            fontsize=12,
        )
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.axis("off")

        # Plot 2: Angle distributions
        # Create subplots for angle distributions - 2 rows, 5 columns
        num_angles = len(angle_cols)
        fig2, axes = plt.subplots(2, 5, figsize=(20, 8))
        axes = axes.flatten()

        for i, angle_col in enumerate(angle_cols):
            if i < len(axes):
                axes[i].hist(
                    df[angle_col].values, bins=30, alpha=0.7, edgecolor="black"
                )
                axes[i].set_xlabel("DSCS Value")
                axes[i].set_ylabel("Number of Samples")
                axes[i].set_title(f"DSCS Distribution on {angle_col}$\degree$")
                axes[i].grid(True, alpha=0.3)

        # Hide empty subplots
        for i in range(num_angles, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        plt.savefig(
            "diffusion_model/angle_distributions.png", dpi=300, bbox_inches="tight"
        )
        plt.show()

        # Save the main PCA plot
        fig.tight_layout()
        fig.savefig(
            "diffusion_model/dataset_pca_analysis.png", dpi=300, bbox_inches="tight"
        )
        fig.show()

        # Print PCA information
        print(f"\nPCA Explained Variance Ratio: {pca.explained_variance_ratio_}")
        print(f"Total Explained Variance: {sum(pca.explained_variance_ratio_):.3f}")

        return pca_result, pca


def main():
    """Main function to run the PCA analysis"""

    # Path to the dataset
    csv_path = "diffusion_model/conditional_csv_datasets/conditional_dataset_6000.csv"

    # Check if file exists
    if not Path(csv_path).exists():
        print(f"Dataset file not found: {csv_path}")
        return

    # Load and analyze the dataset
    df = load_and_analyze_dataset(csv_path)

    # Create PCA visualization
    pca_result, pca = create_pca_visualization(df)

    print("\nPCA analysis completed!")
    print("Plots saved as:")
    print("- diffusion_model/dataset_pca_analysis.png")
    print("- diffusion_model/angle_distributions.png (or feature_distributions.png)")


if __name__ == "__main__":
    main()
