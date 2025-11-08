from pathlib import Path
from typing import List

from dataset_preparation import create_model_dataset
from denoising_diffusion_pytorch_1d import (
    Dataset1D,
    GaussianDiffusion1D,
    Trainer1D,
    Unet1D,
)


class Diffusion_model:
    def __init__(self, milestone: int):
        self.model = Unet1D(dim=32, dim_mults=(1, 2, 4, 8), channels=1)

        self.diffusion = GaussianDiffusion1D(
            self.model, seq_length=32, timesteps=1000, objective="pred_v"
        )

        self.milestone = milestone

    def __dataset_create__(self, dataset_path: Path, angles: List[int]):
        training_dataset = create_model_dataset(dataset_path, angles)
        self.diffusion(training_dataset)
        dataset = Dataset1D(training_dataset)
        return dataset

    def __train__(
        self,
        dataset: Dataset1D,
        train_batch_size: int,
        train_lr: float,
        train_num_steps: int,
        gradient_accumulate_every: int,
        ema_decay: float,
        amp: bool,
        loss_path: Path,
    ):
        self.trainer = Trainer1D(
            self.diffusion,
            dataset=dataset,
            train_batch_size=train_batch_size,
            train_lr=train_lr,
            train_num_steps=train_num_steps,
            gradient_accumulate_every=gradient_accumulate_every,
            ema_decay=ema_decay,
            amp=amp,
        )

        self.trainer.train(path=loss_path, loss_number=self.milestone)
        self.trainer.save(self.milestone)

    def __sample__(self, batch_size: int):
        self.trainer.load(self.milestone)

        sampled_seq = self.diffusion.sample(batch_size=batch_size)

        return sampled_seq


if __name__ == "__main__":
    model = Diffusion_model(milestone=1)
    dataset = model.__dataset_create__(
        Path("diffusion_model/training_dataset"),
        angles=[
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
    model.__train__(
        dataset=dataset,
        train_batch_size=1,
        train_lr=8e-4,
        train_num_steps=10,
        gradient_accumulate_every=2,
        ema_decay=0.995,
        amp=True,
        loss_path=Path("diffusion_model/training_loss"),
    )
    sampled_seq = model.__sample__(batch_size=1)
    print(sampled_seq, sampled_seq.shape)
