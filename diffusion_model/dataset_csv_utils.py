import json
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import torch


def save_dataset_to_csv(
    experiment_dir: Path,
    angles: List[float],
    output_csv_path: Path,
    dataset_type: str = "conditional",
) -> None:
    data_rows = []

    for exp_dir in experiment_dir.iterdir():
        if not exp_dir.is_dir() or not exp_dir.name.startswith("experiment_"):
            continue

        coordinates = json.load(open(exp_dir / "coordinates.json"))
        radiuses = json.load(open(exp_dir / "radiuses.json"))
        dscs_surface = json.load(open(exp_dir / "dscs_surface.json"))
        angles_list = json.load(open(exp_dir / "angles.json"))
        hyper_parameters = json.load(open(exp_dir / "hyperparameters.json"))

        angle_indices = []
        for target_angle in angles:
            try:
                idx = angles_list.index(target_angle)
                angle_indices.append(idx)
            except ValueError:
                raise ValueError(
                    f"Angle {target_angle} not found in experiment data angles"
                )

        flat_coordinates = [coord for point in coordinates for coord in point]

        row_data = {"experiment_id": exp_dir.name, "seed": hyper_parameters["seed"]}

        for i, coord in enumerate(flat_coordinates):
            row_data[f"coord_{i}"] = coord

        for i, radius in enumerate(radiuses):
            row_data[f"radius_{i}"] = radius

        for i, angle_idx in enumerate(angle_indices):
            row_data[f"dscs_angle_{angles[i]}"] = dscs_surface[angle_idx]

        for key, value in hyper_parameters.items():
            if key not in ["seed"]:
                row_data[f"hyperparam_{key}"] = value

        data_rows.append(row_data)

    df = pd.DataFrame(data_rows)
    df.to_csv(output_csv_path, index=False)


def load_dataset_from_csv(
    csv_path: Path, angles: List[float], dataset_type: str = "conditional"
) -> Tuple[torch.Tensor, torch.Tensor]:
    df = pd.read_csv(csv_path)

    coord_cols = [col for col in df.columns if col.startswith("coord_")]
    radius_cols = [col for col in df.columns if col.startswith("radius_")]
    dscs_cols = [col for col in df.columns if col.startswith("dscs_angle_")]

    coord_cols.sort(key=lambda x: int(x.split("_")[1]))
    radius_cols.sort(key=lambda x: int(x.split("_")[1]))
    dscs_cols.sort(key=lambda x: float(x.split("_")[-1]))

    features_list = []
    conditions_list = []

    for _, row in df.iterrows():
        coords = [row[col] for col in coord_cols]
        radiuses = [row[col] for col in radius_cols]

        if dataset_type == "conditional":
            features = coords + radiuses + [0, 0, 0, 0]
            features_tensor = torch.tensor(features, dtype=torch.float32)
            features_tensor = features_tensor.view(
                1, 1, -1
            )  # (batch_size=1, channels=1, sequence_length)

            dscs_values = [row[col] for col in dscs_cols]
            conditions_tensor = torch.tensor(dscs_values, dtype=torch.float32)
            conditions_tensor = conditions_tensor.view(
                1, -1
            )  # (batch_size=1, num_angles)

            features_list.append(features_tensor)
            conditions_list.append(conditions_tensor)

        else:  # unconditional
            dscs_values = [row[col] for col in dscs_cols]
            features = coords + radiuses + dscs_values

            features_tensor = torch.tensor(features, dtype=torch.float32)
            features_tensor = features_tensor.view(
                1, 1, -1
            )  # (batch_size=1, channels=1, sequence_length)

            features_list.append(features_tensor)

    if dataset_type == "conditional":
        dataset = torch.cat(features_list, dim=0)
        conditions = torch.cat(conditions_list, dim=0)
        return dataset, conditions
    else:
        dataset = torch.cat(features_list, dim=0)
        return dataset


def load_conditional_dataset_from_csv(
    csv_path: Path,
) -> Tuple[torch.Tensor, torch.Tensor]:
    df = pd.read_csv(csv_path)

    coord_cols = sorted(
        [col for col in df.columns if col.startswith("coord_")],
        key=lambda x: int(x.split("_")[1]),
    )
    radius_cols = sorted(
        [col for col in df.columns if col.startswith("radius_")],
        key=lambda x: int(x.split("_")[1]),
    )
    dscs_cols = sorted(
        [col for col in df.columns if col.startswith("dscs_angle_")],
        key=lambda x: float(x.split("_")[-1]),
    )

    model_dataset = []
    model_conditions = []

    for _, row in df.iterrows():
        flat_coordinates = [row[col] for col in coord_cols]
        radiuses = [row[col] for col in radius_cols]

        features = flat_coordinates + radiuses + [0] + [0] + [0] + [0]

        step_tensor = torch.tensor(features, dtype=torch.float32)
        step_tensor = step_tensor.view(
            1, 1, -1
        )  # (batch_size=1, channels=1, sequence_length)

        dscs_values = [row[col] for col in dscs_cols]
        conditional_vectors = torch.tensor(dscs_values, dtype=torch.float32)
        conditional_vectors = conditional_vectors.view(
            1, -1
        )  # (batch_size=1, num_angles)

        model_dataset.append(step_tensor)
        model_conditions.append(conditional_vectors)

    dataset = torch.cat(model_dataset, dim=0)  # (batch_size, 1, sequence_length)
    conditions = torch.cat(model_conditions, dim=0)  # (batch_size, num_angles)

    return dataset, conditions


def load_unconditional_dataset_from_csv(csv_path: Path) -> torch.Tensor:
    df = pd.read_csv(csv_path)

    # Extract columns in correct order
    coord_cols = sorted(
        [col for col in df.columns if col.startswith("coord_")],
        key=lambda x: int(x.split("_")[1]),
    )
    radius_cols = sorted(
        [col for col in df.columns if col.startswith("radius_")],
        key=lambda x: int(x.split("_")[1]),
    )
    dscs_cols = sorted(
        [col for col in df.columns if col.startswith("dscs_angle_")],
        key=lambda x: float(x.split("_")[-1]),
    )

    model_dataset = []

    for _, row in df.iterrows():
        flat_coordinates = [row[col] for col in coord_cols]
        radiuses = [row[col] for col in radius_cols]
        dscs_values = [row[col] for col in dscs_cols]

        features = flat_coordinates + radiuses + dscs_values

        step_tensor = torch.tensor(features, dtype=torch.float32)
        step_tensor = step_tensor.view(
            1, 1, -1
        )  # (batch_size=1, channels=1, sequence_length)

        model_dataset.append(step_tensor)

    dataset = torch.cat(model_dataset, dim=0)  # (batch_size, 1, sequence_length)

    return dataset


if __name__ == "__main__":
    experiment_dir = Path("diffusion_model/training_dataset")
    angles = [10, 20, 40, 60, 70, 80, 100, 120, 140, 160]

    save_dataset_to_csv(
        experiment_dir=experiment_dir,
        angles=angles,
        output_csv_path=Path("diffusion_model/conditional_dataset_15000.csv"),
        dataset_type="conditional",
    )

    conditional_dataset, conditional_conditions = load_conditional_dataset_from_csv(
        csv_path=Path("diffusion_model/conditional_dataset_15000.csv")
    )

    print(conditional_dataset.shape)
    print(conditional_conditions.shape)
