import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).parent.parent))

from typing import List, Optional

from conditional_dataset_preparation import create_conditional_model_dataset
from conditional_diffusion_pytorch_1d import (
    Dataset1D,
    GaussianDiffusion1D,
    Trainer1D,
    Unet1D,
)
from dataset_csv_utils import load_conditional_dataset_from_csv


class Diffusion_model:
    def __init__(
        self,
        milestone: Optional[int] = 38,
        use_film: bool = True,
        cond_dim: Optional[int] = 10,
        train_batch_size: int = 16,
        train_lr: float = 4e-4,
        train_num_steps: int = 2000,
        gradient_accumulate_every: int = 2,
        ema_decay: float = 0.995,
        timesteps: int = 1000,
        amp: bool = True,
        angles: List[int] = [0, 10, 20, 40, 60, 80, 100, 120, 140, 160],
        results_folder: Path = Path("diffusion_model/trained_models"),
        dataset_path: Path = Path("diffusion_model/training_dataset"),
        loss_path: Path = Path("diffusion_model/training_loss"),
        model_path: Optional[str] = None,
        dataset_file_path: Optional[
            str
        ] = "diffusion_model/conditional_csv_datasets/conditional_dataset_11000.csv",
    ):
        self.train_batch_size = train_batch_size
        self.train_lr = train_lr
        self.train_num_steps = train_num_steps
        self.gradient_accumulate_every = gradient_accumulate_every
        self.ema_decay = ema_decay
        self.amp = amp
        self.angles = angles
        self.dataset_path = dataset_path
        self.loss_path = loss_path
        self.results_folder = results_folder
        self.timesteps = timesteps
        self.model_path = model_path
        self.dataset_file_path = dataset_file_path

        self.model = Unet1D(
            dim=16,
            init_dim=16,
            dim_mults=(1, 2, 4, 8),
            channels=1,
            cond_dim=cond_dim,
            use_film=use_film,
        )

        self.diffusion = GaussianDiffusion1D(
            self.model, seq_length=16, timesteps=self.timesteps, objective="pred_v"
        )

        self.milestone = milestone

        # training_dataset, conditions = create_conditional_model_dataset(self.dataset_path, self.angles)
        training_dataset, conditions = load_conditional_dataset_from_csv(
            self.dataset_file_path
        )
        self.dataset = Dataset1D(training_dataset, conditions)

        self.trainer = Trainer1D(
            self.diffusion,
            dataset=self.dataset,
            train_batch_size=self.train_batch_size,
            train_lr=self.train_lr,
            train_num_steps=self.train_num_steps,
            gradient_accumulate_every=self.gradient_accumulate_every,
            ema_decay=self.ema_decay,
            results_folder=self.results_folder,
            amp=self.amp,
        )

    def __dataset_create__(self, dataset_path: Path, angles: List[int]):
        training_dataset, conditions = create_conditional_model_dataset(
            dataset_path, angles
        )
        dataset = Dataset1D(training_dataset, conditions)
        return dataset

    def __train__(self):
        self.trainer.train(path=self.loss_path, loss_number=self.milestone)
        self.trainer.save(self.milestone)

    def __load__(self):
        if self.model_path is not None:
            self.trainer.load(self.milestone, model_path=self.model_path)
        else:
            self.trainer.load(self.milestone)

    def __sample__(self, batch_size: int, conditional_vec: torch.Tensor):
        self.__load__()

        sampled_seq = self.diffusion.p_sample_loop(
            shape=(batch_size, 1, 16), cond=conditional_vec
        )

        return sampled_seq

    def __sample_ddim__(self, shape: tuple, conditional_vec: torch.Tensor):
        self.__load__()
        sampled_seq = self.diffusion.ddim_sample(shape=shape, cond=conditional_vec)
        return sampled_seq

    def __visualize_denoising_process__(
        self, shape: tuple, conditional_vec: torch.Tensor, specific_steps=None
    ):
        self.__load__()
        denoising_steps = self.diffusion.visualize_denoising_process(
            shape=shape, cond=conditional_vec, specific_steps=specific_steps
        )
        return denoising_steps


if __name__ == "__main__":
    model = Diffusion_model(milestone=6000, use_film=True)
    model.__train__()
    sampled_seq = model.__sample__(
        batch_size=1,
        conditional_vec=torch.tensor(
            [[0.3, 0.1, 0.05, 0.03, 0.01, 0.04, 0.08, 0.1, 0.09, 0.15]],
            dtype=torch.float32,
        ),
    )
    print(sampled_seq, sampled_seq.shape)
