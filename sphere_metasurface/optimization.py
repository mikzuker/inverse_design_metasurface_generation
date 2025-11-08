import contextlib
import io
import json
import logging
import random
from pathlib import Path

import cmaes
import matplotlib.pyplot as plt
import numpy as np
import ray  
import smuthi.particles as particles
from fitness_function import (
    calculate_loss,
    calculate_spectrum,
    precompute_full_object_spectrum,
    precompute_object_spectrum,
)
from parametrization import Sphere_surface
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("optimization")

# Register custom classes for Ray serialization if needed
try:
    ray.util.register_serialization_context("sphere_metasurface")
except (AttributeError, ImportError):
    # This is fine if we're using a Ray version that doesn't have this feature
    pass


class Optimization(object):
    def __init__(
        self,
        object_to_mimic,
        vacuum_wavelength: float,
        angeles_to_mimic: list,
        side_length: float,
        number_of_cells: int,
        refractive_index: complex,
        iterations: int,
        seed: int,
        num_workers: int = None,  # Added parameter to control number of Ray workers
        use_parallel: bool = True,  # Added parameter to enable/disable parallelization
    ):
        """
        Initialize the optimization object.

        Parameters:
        -----------
        object_to_mimic: Object
            Object to mimic, can be a sphere, cylinder, or list of such objects.
        vacuum_wavelength: float
            Vacuum wavelength of the incident light.
        angeles_to_mimic: list
            List of angles to mimic the spectrum at.
        side_length: float
            Side length of the surface.
        number_of_cells: int
            Number of cells in one dimension.
        refractive_index: complex
            Refractive index of the material.
        iterations: int
            Number of iterations to run the optimization for.
        seed: int
            Random seed for reproducibility.
        num_workers: int, optional
            Number of Ray workers to use for parallelization. If None, Ray will use all available resources.
        use_parallel: bool, optional
            Whether to use Ray for parallelization. If False, computations will be performed sequentially.
        """
        # Store all parameters
        self.object_to_mimic = object_to_mimic
        self.vacuum_wavelength = vacuum_wavelength
        self.angeles_to_mimic = angeles_to_mimic
        self.side_length = side_length
        self.number_of_cells = number_of_cells
        self.refractive_index = refractive_index
        self.iterations = iterations
        self.seed = seed
        self.num_workers = num_workers  # Store the number of workers parameter
        self.use_parallel = use_parallel  # Store whether to use parallelization

        # Create the surface object
        self.surface = Sphere_surface(
            number_of_cells=number_of_cells,
            side_length=side_length,
            reflective_index=refractive_index,
        )

        # Precompute object spectrum once to save computational resources
        self.precomputed_object_spectrum = precompute_object_spectrum(
            object_to_mimic, vacuum_wavelength, angeles_to_mimic
        )

        # Precompute full object spectrum once for plotting and comparisons
        self.precomputed_full_object_spectrum = precompute_full_object_spectrum(
            object_to_mimic, vacuum_wavelength
        )
        print("Object spectra precomputation complete.")

    def extrapolate_params(self, params):
        real_params = []

        for i in range(len(self.surface.squares)):
            square = self.surface.squares[i]
            cell_size = square[1][0] - square[0][0]

            radius = params[i * 3 + 2] * (cell_size / 2)

            x_min = square[0][0] + radius
            x_max = square[1][0] - radius
            x = x_min + params[i * 3] * (x_max - x_min)

            y_min = square[0][1] + radius
            y_max = square[1][1] - radius
            y = y_min + params[i * 3 + 1] * (y_max - y_min)

            real_params.extend([x, y, radius])

        return real_params

    def optimize(self):
        """
        Optimize the sphere surface using CMA-ES algorithm.
        """
        # Initialize Ray only if parallelization is enabled
        if self.use_parallel:
            ray_init_args = {
                "ignore_reinit_error": True,
                # "runtime_env": {"pip": ["numpy", "matplotlib"]}
            }

            if self.num_workers is not None:
                ray_init_args["num_cpus"] = self.num_workers
                logger.info(f"Configuring Ray with {self.num_workers} workers")

            try:
                ray.init(**ray_init_args)
                logger.info("Ray initialized successfully")
            except Exception as e:
                logger.warning(
                    f"Ray initialization warning (this may be ok if already initialized): {str(e)}"
                )
        else:
            logger.info("Parallelization disabled, running in sequential mode")

        self.surface.mesh_generation()
        logger.info(
            f"Starting optimization with {self.number_of_cells}x{self.number_of_cells} cells grid"
        )

        n_spheres = self.number_of_cells**2
        random.seed(self.seed)
        initial_params = [random.uniform(0, 1) for _ in range(3 * n_spheres)]

        cell_size = self.surface.side_length / self.surface.number_of_cells

        # Define a local, non-parallel objective function for sequential execution
        def objective_function_local(params, extrapolate_params_result):
            with (
                contextlib.redirect_stdout(io.StringIO()) as stdout,
                contextlib.redirect_stderr(io.StringIO()) as stderr,
            ):
                try:
                    # Use the already extrapolated parameters to avoid serialization issues
                    real_params = extrapolate_params_result
                    real_x = real_params[::3]
                    real_y = real_params[1::3]
                    real_coordinates_list = [
                        [real_x[i], real_y[i], 0] for i in range(len(real_x))
                    ]

                    surface = Sphere_surface(
                        self.number_of_cells, self.side_length, self.refractive_index
                    )
                    surface.__spheres_add__(
                        coordinates_list=real_coordinates_list,
                        spheres_radius_list=real_params[2::3],
                    )

                    loss_value = calculate_loss(
                        surface,
                        self.object_to_mimic,
                        self.vacuum_wavelength,
                        self.angeles_to_mimic,
                        precomputed_dscs_object=self.precomputed_object_spectrum,
                    )

                    return loss_value
                except Exception as e:
                    error_message = f"Error in objective function: {str(e)}\nSTDOUT: {stdout.getvalue()}\nSTDERR: {stderr.getvalue()}"
                    print(error_message)
                    # If an exception occurs, return a high loss value
                    # This ensures the optimization continues even if a calculation fails
                    return 1e10  # A very high loss value

        # Define the objective function as a remote Ray function with proper error handling
        @ray.remote
        def objective_function_remote(
            params,
            number_of_cells,
            side_length,
            refractive_index,
            extrapolate_params_result,
            object_to_mimic,
            vacuum_wavelength,
            angeles_to_mimic,
            precomputed_object_spectrum,
        ):
            # Use a context manager approach to redirect stdout and stderr
            with (
                contextlib.redirect_stdout(io.StringIO()) as stdout,
                contextlib.redirect_stderr(io.StringIO()) as stderr,
            ):
                try:
                    # Use the already extrapolated parameters to avoid serialization issues
                    real_params = extrapolate_params_result
                    real_x = real_params[::3]
                    real_y = real_params[1::3]
                    real_coordinates_list = [
                        [real_x[i], real_y[i], 0] for i in range(len(real_x))
                    ]

                    surface = Sphere_surface(
                        number_of_cells, side_length, refractive_index
                    )
                    surface.__spheres_add__(
                        coordinates_list=real_coordinates_list,
                        spheres_radius_list=real_params[2::3],
                    )

                    loss_value = calculate_loss(
                        surface,
                        object_to_mimic,
                        vacuum_wavelength,
                        angeles_to_mimic,
                        precomputed_dscs_object=precomputed_object_spectrum,
                    )

                    return loss_value
                except Exception as e:
                    error_message = f"Error in objective function: {str(e)}\nSTDOUT: {stdout.getvalue()}\nSTDERR: {stderr.getvalue()}"
                    print(error_message)
                    # If an exception occurs, return a high loss value
                    # This ensures the optimization continues even if a calculation fails
                    return 1e10  # A very high loss value

        population_size = 70
        logger.info(f"Using population size: {population_size}")

        def objective_function(params):
            real_params = self.extrapolate_params(params)
            real_x = real_params[::3]
            real_y = real_params[1::3]
            real_coordinates_list = [
                [real_x[i], real_y[i], 0] for i in range(len(real_x))
            ]

            surface = Sphere_surface(
                self.number_of_cells, self.side_length, self.refractive_index
            )
            surface.__spheres_add__(
                coordinates_list=real_coordinates_list,
                spheres_radius_list=real_params[2::3],
            )

            loss_value = calculate_loss(
                surface,
                self.object_to_mimic,
                self.vacuum_wavelength,
                self.angeles_to_mimic,
            )

            return loss_value

        population_size = 60
        opts = cmaes.CMA(
            mean=np.array(initial_params),
            sigma=0.1 * cell_size,
            bounds=np.tile([0, 1], (len(initial_params), 1)),
            seed=self.seed,
            population_size=population_size,
        )
        cnt = 0

        max_value, max_params = 100000, []

        pbar = tqdm(range(self.iterations))
        progress = []

        for generation in pbar:
            try:
                solutions = []
                values = []

                # Generate all parameter sets for this generation
                params_list = [opts.ask() for _ in range(population_size)]

                try:
                    # Pre-compute extrapolated params to avoid serialization issues
                    extrapolated_params_list = [
                        self.extrapolate_params(params) for params in params_list
                    ]

                    if self.use_parallel:
                        # Evaluate all parameter sets in parallel using Ray
                        value_refs = [
                            objective_function_remote.remote(
                                params_list[i],
                                self.number_of_cells,
                                self.side_length,
                                self.refractive_index,
                                extrapolated_params_list[i],
                                self.object_to_mimic,
                                self.vacuum_wavelength,
                                self.angeles_to_mimic,
                                self.precomputed_object_spectrum,
                            )
                            for i in range(len(params_list))
                        ]

                        # Wait for all evaluations to complete with a timeout
                        values = ray.get(
                            value_refs, timeout=300
                        )  # 5-minute timeout as a safety measure
                    else:
                        # Evaluate all parameter sets sequentially
                        values = []
                        for i, params in enumerate(params_list):
                            try:
                                loss_value = objective_function_local(
                                    params, extrapolated_params_list[i]
                                )
                                values.append(loss_value)
                            except Exception as inner_e:
                                logger.error(
                                    f"Error in sequential computation for individual {i}: {str(inner_e)}"
                                )
                                values.append(1e10)  # High loss for failed evaluations
                except Exception as e:
                    logger.error(f"Error in parallel execution: {str(e)}")
                    # Fallback: compute sequentially if parallel execution fails
                    logger.warning(
                        "Falling back to sequential computation for this generation"
                    )
                    values = []
                    for i, params in enumerate(params_list):
                        try:
                            loss_value = objective_function_local(
                                params, extrapolated_params_list[i]
                            )
                            values.append(loss_value)
                        except Exception as inner_e:
                            logger.error(
                                f"Error in sequential fallback for individual {i}: {str(inner_e)}"
                            )
                            values.append(1e10)  # High loss for failed evaluations

                # Process results
                for i, (params, value) in enumerate(zip(params_list, values)):
                    if value < max_value:
                        max_value = value
                        max_params = params
                        cnt += 1
                    solutions.append((params, value))

                opts.tell(solutions)
                progress.append(np.around(np.mean(values), 15))

                pbar.set_description(
                    "Processed %s generation\t max %s mean %s"
                    % (
                        generation,
                        np.around(max_value, 15),
                        np.around(np.mean(values), 15),
                    )
                )
            except Exception as generation_error:
                logger.error(
                    f"Error processing generation {generation}: {str(generation_error)}"
                )
                # Continue to next generation if there's an error
                continue

        # Shutdown Ray after optimization is complete if it was used
        if self.use_parallel:
            try:
                ray.shutdown()
                logger.info("Ray shutdown completed")
            except Exception as e:
                logger.warning(f"Ray shutdown warning: {str(e)}")

        results = {
            "params": max_params,
            "optimized_value": max_value,
            "progress": progress,
        }

        experiment_dir = (
            Path("sphere_metasurface/results")
            / f"experiment_{self.side_length}_{self.number_of_cells}_{self.number_of_cells}_{self.refractive_index}_{self.seed}_{self.iterations}"
        )
        experiment_dir.mkdir(parents=True, exist_ok=True)

        # json.dump(results, open(experiment_dir / 'optmization_results.json', 'w'))

        def plot_optimized_structure(results):
            real_params = self.extrapolate_params(results["params"])
            real_x = real_params[::3]
            real_y = real_params[1::3]
            real_coordinates_list = [
                [real_x[i], real_y[i], 0] for i in range(len(real_x))
            ]
            real_radiuses = real_params[2::3]

            surface = Sphere_surface(
                self.number_of_cells, self.side_length, self.refractive_index
            )
            surface.__spheres_add__(
                coordinates_list=real_coordinates_list,
                spheres_radius_list=real_radiuses,
            )
            surface.mesh_generation()

            surface.spheres_plot(
                save_path=experiment_dir / "spheres_surface_projection.pdf"
            )

            json.dump(
                real_coordinates_list, open(experiment_dir / "coordinates.json", "w")
            )
            json.dump(real_radiuses, open(experiment_dir / "radiuses.json", "w"))

        def plot_progress(results):
            plt.figure(figsize=(10, 10))
            generations = range(len(results["progress"]))
            plt.plot(generations, results["progress"])
            plt.title("Optimization progress")
            plt.xlabel("Generation")
            plt.ylabel("Loss value")
            # plt.yscale('log')
            plt.savefig(experiment_dir / "Optimization_progress.pdf")
            plt.close()

        def plot_spectrum(results):
            real_params = self.extrapolate_params(results["params"])
            real_x = real_params[::3]
            real_y = real_params[1::3]
            real_coordinates_list = [
                [real_x[i], real_y[i], 0] for i in range(len(real_x))
            ]
            real_radiuses = real_params[2::3]

            surface = Sphere_surface(
                self.number_of_cells, self.side_length, self.refractive_index
            )
            surface.__spheres_add__(
                coordinates_list=real_coordinates_list,
                spheres_radius_list=real_radiuses,
            )

            plt.figure(figsize=(10, 10))
            # Use the precomputed full object spectrum to save computational resources
            whole_dscs_surface, _ = calculate_spectrum(
                surface,
                self.object_to_mimic,
                self.vacuum_wavelength,
                precomputed_dscs_object=self.precomputed_full_object_spectrum,
            )

            # Use the precomputed full object spectrum directly
            whole_dscs_object = self.precomputed_full_object_spectrum

            surface_array = [np.arange(0, 180, 0.5), whole_dscs_surface]
            object_array = [np.arange(0, 180, 0.5), whole_dscs_object]
            plt.plot(*surface_array, label="surface", linewidth=2)
            plt.plot(*object_array, label="object", linewidth=2)

            angles_indices = np.round(np.degrees(self.angeles_to_mimic)).astype(int)

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

            target_values_object_list = list(target_values_object)
            target_values_surface_list = list(target_values_surface)

            json.dump(
                target_values_object_list,
                open(experiment_dir / "object_target_values.json", "w"),
            )
            json.dump(
                target_values_surface_list,
                open(experiment_dir / "surface_target_values.json", "w"),
            )

            plt.legend()
            plt.xlabel("Angle")
            plt.ylabel("DSCS")
            plt.title("Spectrum")
            plt.grid()
            # plt.yscale('log')
            plt.savefig(experiment_dir / "spectrum.pdf")
            plt.close()

        def save_hyperparameters(self, experiment_dir):
            hyperparameters_file = experiment_dir / "hyperparameters.json"

            # Convert Sphere objects to dictionaries with parameters
            object_params = {
                "radius": float(self.object_to_mimic.cylinder_radius),
                "position": [float(p) for p in self.object_to_mimic.position],
                "height": float(self.object_to_mimic.cylinder_height),
                "euler_angles": [
                    float(angle) for angle in self.object_to_mimic.euler_angles
                ],
                "refractive_index": {
                    "real": float(self.object_to_mimic.refractive_index.real),
                    "imag": float(self.object_to_mimic.refractive_index.imag),
                },
            }

            hyperparameters = {
                "object_type": self.object_to_mimic.__class__.__name__,
                "object_to_mimic": object_params,
                "vacuum_wavelength": float(self.vacuum_wavelength),
                "angeles_to_mimic": [float(angle) for angle in self.angeles_to_mimic],
                "side_length": float(self.side_length),
                "number_of_cells": int(self.number_of_cells),
                "refractive_index": float(self.refractive_index),
                "iterations": int(self.iterations),
                "seed": int(self.seed),
            }

            with open(hyperparameters_file, "w") as f:
                json.dump(hyperparameters, f, indent=4)

        save_hyperparameters(self, experiment_dir)

        return (
            plot_optimized_structure(results),
            plot_progress(results),
            plot_spectrum(results),
        )


if __name__ == "__main__":
    object_to_mimic = particles.FiniteCylinder(
        position=[0, 0, 0],
        refractive_index=2,
        cylinder_radius=5,
        cylinder_height=2,
        euler_angles=[0, 0, 0],
    )

    # Example 1: Use all available CPU cores for parallelization
    optimizer = Optimization(
        object_to_mimic=object_to_mimic,
        vacuum_wavelength=0.04,
        angeles_to_mimic=np.array(
            [
                np.deg2rad(25),
                np.deg2rad(45),
                np.deg2rad(65),
                np.deg2rad(115),
                np.deg2rad(140),
                np.deg2rad(150),
                np.deg2rad(175),
            ]
        ),
        side_length=24.0,
        number_of_cells=3,
        iterations=1,
        refractive_index=10,
        seed=43,
        num_workers=None,  # Use all available cores
    )

    optimized_surface = optimizer.optimize()

    # Example 2: Specify a fixed number of workers (e.g., 4 cores)
    # To use this example, uncomment the code below
    """
    optimizer = Optimization(object_to_mimic=object_to_mimic, 
                        vacuum_wavelength=0.5, 
                        angeles_to_mimic=np.array([np.deg2rad(24), np.deg2rad(70), np.deg2rad(115), np.deg2rad(170)]), 
                        side_length=4.0, 
                        number_of_cells=3, 
                        refractive_index=4+0.1j, 
                        iterations=1500, 
                        seed=43,
                        num_workers=4  # Use exactly 4 cores
                        )

    optimized_surface = optimizer.optimize()
    """

    # Example 3: Run without parallelization (sequential mode)
    # To use this example, uncomment the code below
    """
    optimizer = Optimization(object_to_mimic=object_to_mimic, 
                        vacuum_wavelength=2, 
                        angeles_to_mimic=np.array([np.deg2rad(24), np.deg2rad(70), np.deg2rad(115), np.deg2rad(170)]), 
                        side_length=4.0, 
                        number_of_cells=3, 
                        refractive_index=4+0.1j, 
                        iterations=1500, 
                        seed=43,
                        use_parallel=False  # Disable parallelization completely
                        refractive_index=20, 
                        iterations=500, 
                        seed=3
                        )

    optimized_surface = optimizer.optimize()
    """
