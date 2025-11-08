import json
from pathlib import Path
from typing import List

import torch


def create_conditional_model_dataset(experiment_dir: Path, angles: List[float]):
    model_dataset = []
    model_conditions = []

    for dir in experiment_dir.iterdir():
        if dir.is_dir():
            coordinates = json.load(open(dir / "coordinates.json"))
            radiuses = json.load(open(dir / "radiuses.json"))
            dscs_surface = json.load(open(dir / "dscs_surface.json"))
            angles_list = json.load(open(dir / "angles.json"))
            json.load(open(dir / "hyperparameters.json"))

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

            features = flat_coordinates + radiuses + [0] + [0] + [0] + [0]

            step_tensor = torch.tensor(features)
            step_tensor = step_tensor.view(
                1, 1, -1
            )  # (batch_size=1, channels=1, sequence_length)

            conditional_vectors = torch.tensor([dscs_surface[i] for i in angle_indices])
            conditional_vectors = conditional_vectors.view(
                1, -1
            )  # instead of (1, 1, -1)

            model_dataset.append(step_tensor)
            model_conditions.append(conditional_vectors)

    dataset = torch.cat(model_dataset, dim=0)
    conditions = torch.cat(model_conditions, dim=0)  # [batch, 10]
    return dataset, conditions


if __name__ == "__main__":
    dataset, conditions = create_conditional_model_dataset(
        Path("diffusion_model/training_dataset"),
        [10, 20, 40, 60, 70, 80, 100, 120, 140, 160],
    )
    print(dataset, dataset.shape)
    print(conditions, conditions.shape)
