import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from typing import List

import numpy as np
import torch
from dataset_creation import calculate_field, extrapolate_params

from sphere_metasurface.parametrization import Sphere_surface


class Decoder_diffusional_vector:
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
        self.dscs_generated = self.vector_to_decode[
            2 * self.number_of_cells**2 + self.number_of_cells**2 :
        ]

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
        dscs_surface = calculate_field(
            spheres_surface=sphere_surface,
            vacuum_wavelength=self.vacuum_wavelength,
            polar_angle=self.polar_angle,
            azimuthal_angle=self.azimuthal_angle,
            polarization=self.polarization,
        )

        angles_indices = []
        all_angles = np.arange(0, 180, 0.5)
        for angle in self.angles:
            idx = np.argmin(np.abs(all_angles - angle))
            angles_indices.append(idx)
        self.dscs_surface = [dscs_surface[i] for i in angles_indices]

    def compute_loss(self):
        loss_mpe = (100 / len(self.dscs_surface)) * np.abs(
            np.sum(
                [
                    (self.dscs_surface[i] - self.dscs_generated[i])
                    / self.dscs_surface[i]
                    for i in range(len(self.dscs_surface))
                ]
            )
        )
        return loss_mpe


if __name__ == "__main__":
    vector_to_decode = torch.tensor(
        [
            [
                [
                    0.4210,
                    0.4459,
                    0.1909,
                    0.0572,
                    0.2629,
                    0.1530,
                    0.0669,
                    0.4242,
                    0.3791,
                    0.0211,
                    0.0040,
                    0.3807,
                    0.2169,
                    0.2768,
                    0.3678,
                    0.3417,
                    0.2477,
                    0.5718,
                    0.5150,
                    0.4732,
                    0.2740,
                    0.4963,
                    0.3700,
                    0.5132,
                    0.3638,
                    0.0888,
                    0.4749,
                    0.3749,
                    0.5372,
                    0.2344,
                    0.4101,
                    0.1653,
                ]
            ]
        ]
    )
    decoder = Decoder_diffusional_vector(
        vector_to_decode,
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
        number_of_cells=2,
        side_length=10,
        reflective_index=2,
        vacuum_wavelength=1,
    )
    dscs_surface = decoder.compute_sphere_surface()

    loss = decoder.compute_loss()

    print(loss)
