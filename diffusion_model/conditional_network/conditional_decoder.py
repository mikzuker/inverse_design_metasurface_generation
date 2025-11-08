import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
sys.path.append("/workspace/diffusion_model")
sys.path.append("/workspace")
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate
import torch
from conditional_model import Diffusion_model
from dataset_creation import calculate_field, extrapolate_params

from sphere_metasurface.parametrization import Sphere_surface


class Decoder_conditional_diffusion_vector:
    def __init__(
        self,
        vector_to_decode: torch.Tensor,
        angles: List[float],
        number_of_cells: int,
        side_length: float,
        reflective_index: complex,
        vacuum_wavelength: float,
        polar_angle: float = np.pi,  # angle in radians, pi == from top
        azimuthal_angle: float = 0,  # angle in radians, 0 == x-axis
        polarization: int = 0,  # 0 for TE, 1 for TM polarization
        conditional_dscs_surface: List[float] = None,
        denoising_steps: Optional[List[Tuple[int, torch.Tensor]]] = None,
    ):
        self.vector_to_decode = vector_to_decode.tolist()[0][0]
        self.angles = angles
        self.number_of_cells = number_of_cells
        self.side_length = side_length
        self.reflective_index = reflective_index
        self.vacuum_wavelength = vacuum_wavelength
        self.polar_angle = polar_angle
        self.azimuthal_angle = azimuthal_angle
        self.polarization = polarization
        self.conditional_dscs_surface = conditional_dscs_surface
        self.denoising_steps = denoising_steps

    def compute_sphere_surface(self):
        sphere_surface = Sphere_surface(
            number_of_cells=self.number_of_cells,
            side_length=self.side_length,
            reflective_index=self.reflective_index,
        )
        sphere_surface.mesh_generation()

        sphere_coordinates_01 = self.vector_to_decode[: 2 * self.number_of_cells**2]
        sphere_radiuses_01 = self.vector_to_decode[
            2 * self.number_of_cells**2 : 2 * self.number_of_cells**2
            + self.number_of_cells**2
        ]

        parameters_01 = []
        for i in range(len(sphere_radiuses_01)):
            parameters_01.append(sphere_coordinates_01[2 * i])
            parameters_01.append(sphere_coordinates_01[2 * i + 1])
            parameters_01.append(sphere_radiuses_01[i])

        real_params = extrapolate_params(parameters_01, sphere_surface)
        real_x = real_params[::3]
        real_y = real_params[1::3]
        real_radiuses = real_params[2::3]
        real_coordinates = [[real_x[i], real_y[i], 0] for i in range(len(real_x))]

        sphere_surface.__spheres_add__(real_radiuses, real_coordinates)
        self.dscs_surface = calculate_field(
            spheres_surface=sphere_surface,
            vacuum_wavelength=self.vacuum_wavelength,
            polar_angle=self.polar_angle,
            azimuthal_angle=self.azimuthal_angle,
            polarization=self.polarization,
        )
        self.sphere_surface = sphere_surface
        self.angles_indices = []
        self.all_angles = np.arange(0, 180, 0.5)
        for angle in self.angles:
            idx = np.argmin(np.abs(self.all_angles - angle))
            self.angles_indices.append(idx)
        self.dscs_surface_points = [self.dscs_surface[i] for i in self.angles_indices]

    def compute_dscs_surface_from_denoising_steps(self):
        self.dscs_surface_steps = []
        self.steps_numbers = []
        self.denoising_surfaces = []

        for i in range(len(self.denoising_steps)):
            self.steps_numbers.append(self.denoising_steps[i][0])
            # Use denormalized values (already positive)
            denormalized_vector = self.denoising_steps[i][1].tolist()[0][0]
            self.dscs_surface_steps.append(denormalized_vector)

            sphere_surface = Sphere_surface(
                number_of_cells=self.number_of_cells,
                side_length=self.side_length,
                reflective_index=self.reflective_index,
            )
            sphere_surface.mesh_generation()

            # Use denormalized values for coordinates and radii
            sphere_coordinates_01 = denormalized_vector[: 2 * self.number_of_cells**2]
            sphere_radiuses_01 = denormalized_vector[
                2 * self.number_of_cells**2 : 2 * self.number_of_cells**2
                + self.number_of_cells**2
            ]

            parameters_01 = []
            for j in range(len(sphere_radiuses_01)):
                parameters_01.append(sphere_coordinates_01[2 * j])
                parameters_01.append(sphere_coordinates_01[2 * j + 1])
                parameters_01.append(sphere_radiuses_01[j])

            real_params = extrapolate_params(parameters_01, sphere_surface)
            real_x = real_params[::3]
            real_y = real_params[1::3]
            real_radiuses = real_params[2::3]
            real_coordinates = [[real_x[j], real_y[j], 0] for j in range(len(real_x))]

            sphere_surface.__spheres_add__(real_radiuses, real_coordinates)

            self.denoising_surfaces.append(
                calculate_field(
                    spheres_surface=sphere_surface,
                    vacuum_wavelength=self.vacuum_wavelength,
                    polar_angle=self.polar_angle,
                    azimuthal_angle=self.azimuthal_angle,
                    polarization=self.polarization,
                )
            )

    def visualize_denoising_spectrum_evolution(self):
        """
        Visualizes the evolution of the spectrum at each denoising step
        """
        if self.denoising_steps is None:
            print("No denoising steps data!")
            return

        fig, ax = plt.subplots(1, 1, figsize=(15, 8))

        colors = plt.cm.viridis(np.linspace(0, 1, len(self.denoising_steps)))

        for i, (step, tensor) in enumerate(self.denoising_steps):
            denormalized_vector = tensor.tolist()[0][0]
            ax.plot(
                range(len(denormalized_vector)),
                denormalized_vector,
                color=colors[i],
                alpha=0.7,
                linewidth=2,
                label=f"Step {step}",
            )

        ax.set_xlabel("Parameter Index")
        ax.set_ylabel("Value (denormalized)")
        ax.set_title("Evolution of Parameter Vector During Denoising")
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        plt.tight_layout()

        output_path = Path(__file__).parent.parent / "denoising_spectrum_evolution.pdf"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved in: {output_path}")

        plt.show()

    def compute_loss(self):
        loss_mpe = (100 / len(self.dscs_surface_points)) * np.abs(
            np.sum(
                [
                    (self.dscs_surface_points[i] - self.conditional_dscs_surface[i])
                    / self.dscs_surface_points[i]
                    for i in range(len(self.dscs_surface_points))
                ]
            )
        )
        return loss_mpe

    def plot_dscs(self):
        output_path = Path(__file__).parent.parent

        f = scipy.interpolate.interp1d(
            self.angles,
            self.conditional_dscs_surface,
            kind="cubic",
            bounds_error=False,
            fill_value="interpolate",
        )

        plt.figure(figsize=(10, 5))

        plt.plot(self.all_angles, self.dscs_surface, label="Generated Surface", lw=2)

        valid_mask = (self.all_angles >= min(self.angles)) & (
            self.all_angles <= max(self.angles)
        )
        valid_angles = self.all_angles[valid_mask]
        if len(valid_angles) > 0:
            plt.plot(
                valid_angles,
                f(valid_angles),
                label="Interpolated (Real-like) DSCS",
                lw=2,
            )

        # if self.denoising_steps is not None:
        #     self.compute_dscs_surface_from_denoising_steps()
        #     for step, surface in zip(self.steps_numbers, self.denoising_surfaces):
        #         plt.plot(self.all_angles, surface, label=f'Denoising Step {step}', lw=2)

        plt.scatter(self.angles, self.dscs_surface_points, label="Generated DSCS")
        plt.scatter(
            self.angles, self.conditional_dscs_surface, label="Conditional DSCS"
        )

        plt.yscale("log")
        plt.xlabel("Angle, deg")
        plt.ylabel("DSCS, relative units")
        plt.grid(True)
        plt.legend()
        plt.title("DSCS Surface, Generated & Conditional")
        plt.savefig(output_path / "dscs_surface.pdf")

        self.sphere_surface.spheres_plot(save_path=output_path / "sphere_surface.pdf")


if __name__ == "__main__":
    for j in range(38, 39, 1):
        milestone = j
        model = Diffusion_model(
            milestone=milestone,
            use_film=True,
            results_folder=Path(__file__).parent.parent
            / "Conditional_Model_16_2_4e-6_11000_20000",
            timesteps=1000,
            model_path=Path(__file__).parent.parent
            / "Conditional_Model_16_2_4e-6_11000_20000/model.pt",
        )
        output_file = Path(__file__).parent.parent / "try_out.txt"

        with open(output_file, "w") as f:
            for i in range(1):
                print(f"Generation {i + 1}/1...")
                vector_to_decode = model.__sample__(
                    batch_size=1,
                    conditional_vec=torch.tensor(
                        [
                            [
                                0.2440047233652012,
                                0.13373316545853187,
                                0.02634935177036332,
                                0.01681613001080275,
                                0.014719126871156401,
                                0.004718059458417722,
                                0.021446479244308474,
                                0.027422508550410247,
                                0.05498299796079197,
                                0.8492591641224932,
                            ]
                        ]
                    ),
                )
                print(vector_to_decode)

                decoder = Decoder_conditional_diffusion_vector(
                    vector_to_decode,
                    angles=[10, 20, 40, 60, 70, 80, 100, 120, 140, 160],
                    number_of_cells=2,
                    side_length=10,
                    reflective_index=2,
                    vacuum_wavelength=1,
                    conditional_dscs_surface=[
                        0.2440047233652012,
                        0.13373316545853187,
                        0.02634935177036332,
                        0.01681613001080275,
                        0.014719126871156401,
                        0.004718059458417722,
                        0.021446479244308474,
                        0.027422508550410247,
                        0.05498299796079197,
                        0.8492591641224932,
                    ],
                )

                dscs_surface = decoder.compute_sphere_surface()

                loss = decoder.compute_loss()
                f.write(f"{loss}\n")
                print(f"Loss: {loss}")
                decoder.plot_dscs()

        print(f"Results saved in: {output_file}")
