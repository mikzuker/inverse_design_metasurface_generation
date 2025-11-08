import os
import sys

import numpy as np
import smuthi.initial_field
import smuthi.layers
import smuthi.particles
import smuthi.postprocessing.far_field as ff
import smuthi.simulation

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from pathlib import Path

import matplotlib.pyplot as plt
import smuthi.utility.logging as log

from sphere_metasurface.parametrization import Sphere_surface


def calculate_field(
    spheres_surface,
    vacuum_wavelength: float,
    polar_angle: float = np.pi,  # angle in radians, pi == from top
    azimuthal_angle: float = 0,  # angle in radians, 0 == x-axis
    polarization: int = 0,  # 0 for TE, 1 for TM polarization
):
    layers = smuthi.layers.LayerSystem()

    initial_field = smuthi.initial_field.PlaneWave(
        vacuum_wavelength=vacuum_wavelength,
        polar_angle=polar_angle,
        azimuthal_angle=azimuthal_angle,
        polarization=polarization,
    )

    # Create particle lists
    surface_particles = spheres_surface.spheres

    # Set parameters for all particles
    for particle in surface_particles:
        particle.l_max = 5  # multipolar order
        particle.m_max = 5  # azimuthal order

    # Create and run simulation for object
    simulation_surface = smuthi.simulation.Simulation(
        layer_system=layers,
        particle_list=surface_particles,
        initial_field=initial_field,
        log_to_terminal=False,
    )
    with log.LoggerMuted():
        simulation_surface.run()

    whole_far_field_surface = ff.scattered_far_field(
        vacuum_wavelength=vacuum_wavelength,
        particle_list=surface_particles,
        layer_system=layers,
    )

    whole_dscs_surface = (
        np.sum(whole_far_field_surface.azimuthal_integral(), axis=0) * np.pi / 180
    )

    return whole_dscs_surface


def extrapolate_params(params, surface):
    real_params = []
    surface.number_of_cells**2

    for i in range(len(surface.squares)):
        square = surface.squares[i]
        cell_size = square[1][0] - square[0][0]  # cell size

        radius = params[i * 3 + 2] * (cell_size / 2)

        x_min = square[0][0] + radius
        x_max = square[1][0] - radius
        x = x_min + params[i * 3] * (x_max - x_min)

        y_min = square[0][1] + radius
        y_max = square[1][1] - radius
        y = y_min + params[i * 3 + 1] * (y_max - y_min)

        real_params.extend([x, y, radius])

    return real_params


def generate_surface(
    seed: int,
    number_of_cells: int,
    side_length: float,
    reflective_index: complex,
    vacuum_wavelength: float,
    polar_angle: float = np.pi,  # angle in radians, pi == from top
    azimuthal_angle: float = 0,  # angle in radians, 0 == x-axis
    polarization: int = 0,  # 0 for TE, 1 for TM polarization
    experiment_dir: Path = Path("diffusion_model/training_dataset"),
):
    np.random.seed(seed)

    spheres_surface = Sphere_surface(number_of_cells, side_length, reflective_index)
    spheres_surface.mesh_generation()

    # spheres_radius_list = np.random.uniform(1e-2, side_length/2 - 1e-3, number_of_cells**2)
    coordinates_list_01 = np.array(
        [np.random.uniform(0, 1) for _ in range(3 * number_of_cells**2)]
    )
    x = coordinates_list_01[::3]
    y = coordinates_list_01[1::3]
    coordinates_01 = [[x[i], y[i]] for i in range(len(x))]
    radiuses_01 = coordinates_list_01[2::3]

    real_params = extrapolate_params(coordinates_list_01, spheres_surface)
    real_x = real_params[::3]
    real_y = real_params[1::3]
    real_coordinates_list = [[real_x[i], real_y[i], 0] for i in range(len(real_x))]
    real_radiuses = real_params[2::3]

    spheres_surface.__spheres_add__(
        coordinates_list=real_coordinates_list, spheres_radius_list=real_radiuses
    )

    angles = np.arange(0, 180, 0.5)
    dscs_surface = calculate_field(
        spheres_surface, vacuum_wavelength, polar_angle, azimuthal_angle, polarization
    )

    experiment_dir = experiment_dir / f"experiment_{seed}"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    spheres_surface.spheres_plot(
        save_path=experiment_dir / "spheres_surface_projection.pdf"
    )

    # json.dump(coordinates_list_01.tolist(), open(experiment_dir / 'parameters.json', 'w'))
    json.dump(coordinates_01, open(experiment_dir / "coordinates.json", "w"))
    json.dump(radiuses_01.tolist(), open(experiment_dir / "radiuses.json", "w"))
    json.dump(
        real_coordinates_list, open(experiment_dir / "real_coordinates.json", "w")
    )
    json.dump(real_radiuses, open(experiment_dir / "real_radiuses.json", "w"))
    json.dump(dscs_surface.tolist(), open(experiment_dir / "dscs_surface.json", "w"))
    json.dump(angles.tolist(), open(experiment_dir / "angles.json", "w"))

    plt.figure(figsize=(10, 10))

    surface_array = [angles, dscs_surface]
    plt.plot(*surface_array, label="surface", linewidth=2)
    plt.yscale("log")
    plt.legend()
    plt.xlabel("Angle")
    plt.ylabel("DSCS")
    plt.title("Spectrum")
    plt.grid()
    plt.savefig(experiment_dir / "spectrum.pdf")
    plt.close()

    hyperparameters_file = experiment_dir / "hyperparameters.json"

    hyperparameters = {
        "object_type": spheres_surface.__class__.__name__,
        "vacuum_wavelength": float(vacuum_wavelength),
        "side_length": float(side_length),
        "number_of_cells": int(number_of_cells),
        "refractive_index": float(reflective_index),
        "polar_angle": float(polar_angle),
        "azimuthal_angle": float(azimuthal_angle),
        "polarization": int(polarization),
        "seed": int(seed),
    }

    with open(hyperparameters_file, "w") as f:
        json.dump(hyperparameters, f, indent=4)


if __name__ == "__main__":
    for i in range(0, 1):
        generate_surface(
            seed=i,
            number_of_cells=2,
            side_length=10,
            reflective_index=2,
            vacuum_wavelength=1,
            polar_angle=np.pi,
            azimuthal_angle=0,
            polarization=0,
        )
