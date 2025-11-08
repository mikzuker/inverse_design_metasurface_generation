import matplotlib.pyplot as plt
import numpy as np
import smuthi.initial_field
import smuthi.layers
import smuthi.particles
import smuthi.postprocessing.graphical_output as go
import smuthi.simulation


class Sphere_surface(object):
    """
    Class for parametrization of sphere surface
    """

    def __init__(
        self, number_of_cells: int, side_length: float, reflective_index: complex
    ):
        """
        Initialize the sphere surface
        """
        self.number_of_cells = number_of_cells
        self.side_length = side_length
        self.reflective_index = reflective_index

    def mesh_generation(self):
        """
        Mesh generation for the sphere surface
        """
        self.squares = []
        for i in range(self.number_of_cells):
            for j in range(self.number_of_cells):
                x_1 = (i / self.number_of_cells) * self.side_length
                x_2 = ((i + 1) / self.number_of_cells) * self.side_length
                y_1 = (j / self.number_of_cells) * self.side_length
                y_2 = ((j + 1) / self.number_of_cells) * self.side_length

                square = [(x_1, y_1), (x_2, y_2)]
                self.squares.append(square)

    def __spheres_add__(self, spheres_radius_list: list, coordinates_list: list):
        """
        Add spheres to each cell of the mesh using SMUTHI
        """
        self.spheres = []

        for i in range(self.number_of_cells**2):
            sphere = smuthi.particles.Sphere(
                position=coordinates_list[i],
                refractive_index=self.reflective_index,
                radius=spheres_radius_list[i],
                l_max=3,
            )
            self.spheres.append(sphere)

    def spheres_plot(self, save_path: str):
        """
        Plot the spheres in 2D projection
        """
        fig = plt.figure(figsize=(10, 10))

        for sphere in self.spheres:
            x, y = sphere.position[0], sphere.position[1]
            radius = sphere.radius
            circle = plt.Circle((x, y), radius, fill=True, color="blue")
            plt.gca().add_patch(circle)

        for square in self.squares:
            x_coords = [point[0] for point in square]
            y_coords = [point[1] for point in square]
            plt.plot(x_coords, y_coords, "o", color="black")

        plt.plot(0, self.side_length, "o", color="black")  # top left point
        plt.plot(self.side_length, 0, "o", color="black")  # bottom right point

        plt.axis("equal")
        plt.xlabel("X, relative units")
        plt.ylabel("Y, relative units")
        plt.grid(True)
        plt.title("Spheres surface projection")
        plt.savefig(save_path)
        plt.close()
        return fig


if __name__ == "__main__":
    surface = Sphere_surface(
        number_of_cells=3, side_length=3.0, reflective_index=complex(2.0, 0.0)
    )

    surface.mesh_generation()

    coordinates_list = [
        [0.3, 0.5, 0],
        [1.5, 0.4, 0],
        [2.2, 0.5, 0],
        [0.7, 1.5, 0],
        [1.5, 1.4, 0],
        [2.55, 1.5, 0],
        [0.45, 2.45, 0],
        [1.6, 2.7, 0],
        [2.5, 2.4, 0],
    ]

    spheres_radius_list = [0.1] * 9

    surface.__spheres_add__(spheres_radius_list, coordinates_list)

    surface.spheres_plot(
        save_path="/workspace/sphere_metasurface/plots/spheres_surface_projection.png"
    )

    layers = smuthi.layers.LayerSystem()

    plane_wave = smuthi.initial_field.PlaneWave(
        vacuum_wavelength=0.5, polar_angle=np.pi, azimuthal_angle=0, polarization=0
    )

    simulation = smuthi.simulation.Simulation(
        layer_system=layers,
        particle_list=surface.spheres,
        initial_field=plane_wave,
        length_unit="nm",
    )

    simulation.run()

    go.show_scattered_far_field(
        simulation,
        show_plots=True,
        show_opts=[{"label": "scattered_far_field"}],
        save_plots=True,
        save_opts=None,
        save_data=False,
        data_format="hdf5",
        outputdir="/workspace/sphere_metasurface/plots/",
        flip_downward=True,
        split=True,
        log_scale=False,
        polar_angles="default",
        azimuthal_angles="default",
        angular_resolution=None,
    )
