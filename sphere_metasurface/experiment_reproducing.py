import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import smuthi.particles as particles
from fitness_function import calculate_spectrum
from parametrization import Sphere_surface


def reproduce_results(experiment_path, reproduce_dir):
    """
    Reproduces optimization results from saved files

    Args:
        experiment_path (str): Path to the directory with experiment results
    """
    experiment_dir = Path(experiment_path)

    with open(experiment_dir / "hyperparameters.json", "r") as f:
        hyperparameters = json.load(f)

    with open(experiment_dir / "coordinates.json", "r") as f:
        coordinates = json.load(f)
    with open(experiment_dir / "radiuses.json", "r") as f:
        radiuses = json.load(f)

    object_params = hyperparameters["object_to_mimic"]
    if isinstance(object_params, list):
        object_to_mimic = [
            particles.FiniteCylinder(
                cylinder_radius=obj["radius"],
                position=obj["position"],
                cylinder_height=obj["height"],
                refractive_index=complex(
                    obj["refractive_index"]["real"], obj["refractive_index"]["imag"]
                ),
                euler_angles=obj["euler_angles"],
            )
            if "height" in obj
            else particles.Sphere(
                radius=obj["radius"],
                position=obj["position"],
                refractive_index=complex(
                    obj["refractive_index"]["real"], obj["refractive_index"]["imag"]
                ),
            )
            for obj in object_params
        ]
    else:
        if "height" in object_params:
            object_to_mimic = particles.FiniteCylinder(
                cylinder_radius=object_params["radius"],
                position=object_params["position"],
                cylinder_height=object_params["height"],
                refractive_index=complex(
                    object_params["refractive_index"]["real"],
                    object_params["refractive_index"]["imag"],
                ),
                euler_angles=object_params["euler_angles"],
            )
        else:
            object_to_mimic = particles.Sphere(
                radius=object_params["radius"],
                position=object_params["position"],
                refractive_index=complex(
                    object_params["refractive_index"]["real"],
                    object_params["refractive_index"]["imag"],
                ),
            )

    def plot_geometry():
        surface = Sphere_surface(
            number_of_cells=hyperparameters["number_of_cells"],
            side_length=hyperparameters["side_length"],
            reflective_index=complex(hyperparameters["refractive_index"], 0),
        )
        surface.mesh_generation()
        surface.__spheres_add__(
            coordinates_list=coordinates, spheres_radius_list=radiuses
        )
        surface.spheres_plot(
            save_path=reproduce_dir / "spheres_surface_projection_reproduced.pdf"
        )
        return surface

    def plot_spectrum(surface):
        whole_dscs_surface, whole_dscs_object = calculate_spectrum(
            spheres_surface=surface,
            object=object_to_mimic,
            vacuum_wavelength=hyperparameters["vacuum_wavelength"],
        )

        plt.figure(figsize=(10, 10))
        angles = np.arange(0, 180, 0.5)

        plt.plot(angles, whole_dscs_surface, label="surface", linewidth=2)
        plt.plot(angles, whole_dscs_object, label="object", linewidth=2)

        angles_indices = np.round(
            np.degrees(hyperparameters["angeles_to_mimic"])
        ).astype(int)
        target_values_object = whole_dscs_object[
            [angles_indices[i] * 2 for i in range(len(angles_indices))]
        ]
        target_values_surface = whole_dscs_surface[
            [angles_indices[i] * 2 for i in range(len(angles_indices))]
        ]

        plt.scatter(
            angles_indices,
            target_values_object,
            color="red",
            s=100,
            label="target angles object",
        )
        plt.scatter(
            angles_indices,
            target_values_surface,
            color="blue",
            s=100,
            label="target angles surface",
        )

        plt.legend()
        plt.xlabel("Angle")
        plt.ylabel("DSCS")
        plt.title("Spectrum")
        plt.grid(True)
        plt.savefig(reproduce_dir / "spectrum_reproduced.pdf")
        plt.close()

    surface = plot_geometry()
    plot_spectrum(surface)


if __name__ == "__main__":
    experiment_path = "sphere_metasurface/results/experiment_7.0_3_3_10_43_1"
    reproduce_dir = (
        Path("sphere_metasurface/reproduced_experiments") / f"experiment_{2}"
    )
    reproduce_dir.mkdir(parents=True, exist_ok=True)

    reproduce_results(experiment_path, reproduce_dir)
