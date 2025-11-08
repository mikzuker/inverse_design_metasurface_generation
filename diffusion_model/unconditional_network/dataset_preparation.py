import json
from pathlib import Path
from typing import List

import torch


def create_model_dataset(experiment_dir: Path, angles: List[float]):
    model_dataset = []

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

            features = (
                flat_coordinates + radiuses + [dscs_surface[i] for i in angle_indices]
            )

            step_tensor = torch.tensor(features)
            step_tensor = step_tensor.view(
                1, 1, -1
            )  # (batch_size=1, channels=1, sequence_length)
            model_dataset.append(step_tensor)

    dataset = torch.cat(model_dataset, dim=0)
    return dataset


if __name__ == "__main__":
    dataset = create_model_dataset(
        Path("diffusion_model/training_dataset"),
        [
            0,
            5,
            10,
            15,
            20,
            25,
            30,
            35,
            40,
            50,
            60,
            70,
            80,
            90,
            100,
            110,
            120,
            130,
            140,
            150,
        ],
    )
    print(dataset, dataset.shape)
