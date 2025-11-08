import numpy as np
import smuthi.initial_field
import smuthi.layers
import smuthi.particles
import smuthi.postprocessing.far_field as ff
import smuthi.simulation
import smuthi.utility.logging as log
from parametrization import Sphere_surface


def calculate_loss(
    spheres_surface,
    object,
    vacuum_wavelength: float,
    angles_to_mimic: list,
    polar_angle: float = np.pi,  # angle in radians, pi == from top
    azimuthal_angle: float = 0,  # angle in radians, 0 == x-axis
    polarization: int = 0,  # 0 for TE, 1 for TM polarization
    precomputed_dscs_object=None,
):
    """
    Calculate loss value for the given spheres surface, object, initial field, and angle to mimicking
    If precomputed_dscs_object is provided, it will be used instead of recalculating object spectrum
    """
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

    if precomputed_dscs_object is None:
        # Only calculate object spectrum if not precomputed
        object_particles = object

        for particle in object_particles:
            particle.l_max = 5  # multipolar order
            particle.m_max = 5  # azimuthal order

        # Create and run simulation for object
        simulation_object = smuthi.simulation.Simulation(
            layer_system=layers,
            particle_list=object_particles,
            initial_field=initial_field,
            log_to_terminal=False,
        )
        with log.LoggerMuted():
            simulation_object.run()

        far_field_object = ff.scattered_far_field(
            vacuum_wavelength=vacuum_wavelength,
            particle_list=object_particles,
            layer_system=layers,
            polar_angles=np.array(angles_to_mimic),
        )

        dscs_object = (
            np.sum(far_field_object.azimuthal_integral(), axis=0) * np.pi / 180
        )
    else:
        # Use precomputed object spectrum
        dscs_object = precomputed_dscs_object

    # Create and run simulation for surface
    simulation_surface = smuthi.simulation.Simulation(
        layer_system=layers,
        particle_list=surface_particles,
        initial_field=initial_field,
        log_to_terminal=False,
    )
    with log.LoggerMuted():
        simulation_surface.run()

    far_field_surface = ff.scattered_far_field(
        vacuum_wavelength=vacuum_wavelength,
        particle_list=surface_particles,
        layer_system=layers,
        polar_angles=np.array(angles_to_mimic),
    )

    dscs_surface = np.sum(far_field_surface.azimuthal_integral(), axis=0) * np.pi / 180

    loss_value = (
        100
        / len(angles_to_mimic)
        * np.sum(np.abs(dscs_object - dscs_surface) / dscs_object)
    )
    return loss_value


def precompute_object_spectrum(
    object,
    vacuum_wavelength: float,
    angles_to_mimic: list,
    polar_angle: float = np.pi,  # angle in radians, pi == from top
    azimuthal_angle: float = 0,  # angle in radians, 0 == x-axis
    polarization: int = 0,  # 0 for TE, 1 for TM polarization
):
    """
    Precompute the object spectrum for the given angles to mimic
    """
    layers = smuthi.layers.LayerSystem()

    initial_field = smuthi.initial_field.PlaneWave(
        vacuum_wavelength=vacuum_wavelength,
        polar_angle=polar_angle,
        azimuthal_angle=azimuthal_angle,
        polarization=polarization,
    )

    # Create particle list
    object_particles = [object]

    # Set parameters for all particles
    for particle in object_particles:
        particle.l_max = 5  # multipolar order
        particle.m_max = 5  # azimuthal order

    # Create and run simulation for object
    simulation_object = smuthi.simulation.Simulation(
        layer_system=layers,
        particle_list=object_particles,
        initial_field=initial_field,
        log_to_terminal=False,
    )
    with log.LoggerMuted():
        simulation_object.run()

    far_field_object = ff.scattered_far_field(
        vacuum_wavelength=vacuum_wavelength,
        particle_list=object_particles,
        layer_system=layers,
        polar_angles=np.array(angles_to_mimic),
    )

    dscs_object = np.sum(far_field_object.azimuthal_integral(), axis=0) * np.pi / 180

    return dscs_object


def precompute_full_object_spectrum(
    object,
    vacuum_wavelength: float,
    polar_angle: float = np.pi,  # angle in radians, pi == from top
    azimuthal_angle: float = 0,  # angle in radians, 0 == x-axis
    polarization: int = 0,  # 0 for TE, 1 for TM polarization
):
    """
    Precompute the full object spectrum for all angles (0-180 degrees)
    """
    layers = smuthi.layers.LayerSystem()

    initial_field = smuthi.initial_field.PlaneWave(
        vacuum_wavelength=vacuum_wavelength,
        polar_angle=polar_angle,
        azimuthal_angle=azimuthal_angle,
        polarization=polarization,
    )

    # Create particle list
    object_particles = [object]

    # Set parameters for all particles
    for particle in object_particles:
        particle.l_max = 5  # multipolar order
        particle.m_max = 5  # azimuthal order

    # Create and run simulation for object
    simulation_object = smuthi.simulation.Simulation(
        layer_system=layers,
        particle_list=object_particles,
        initial_field=initial_field,
        log_to_terminal=False,
    )
    with log.LoggerMuted():
        simulation_object.run()

    # Get the full spectrum
    whole_far_field_object = ff.scattered_far_field(
        vacuum_wavelength=vacuum_wavelength,
        particle_list=object_particles,
        layer_system=layers,
    )

    whole_dscs_object = (
        np.sum(whole_far_field_object.azimuthal_integral(), axis=0) * np.pi / 180
    )

    return whole_dscs_object


def calculate_spectrum(
    spheres_surface,
    object,
    vacuum_wavelength: float,
    polar_angle: float = np.pi,  # angle in radians, pi == from top
    azimuthal_angle: float = 0,  # angle in radians, 0 == x-axis
    polarization: int = 0,  # 0 for TE, 1 for TM polarization
    precomputed_dscs_object=None,
):
    """
    Calculate spectrum for the given spheres surface and object.
    If precomputed_dscs_object is provided, it will skip calculating object spectrum.
    """
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

    # Create and run simulation for surface
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

    if precomputed_dscs_object is not None:
        # Use precomputed object spectrum if available
        # Note: This assumes the precomputed spectrum covers the full angle range (0-180 degrees)
        whole_dscs_object = precomputed_dscs_object
    else:
        # Calculate object spectrum if not precomputed
        object_particles = [object]

        for particle in object_particles:
            particle.l_max = 5  # multipolar order
            particle.m_max = 5  # azimuthal order

        # Create and run simulation for object
        simulation_object = smuthi.simulation.Simulation(
            layer_system=layers,
            particle_list=object_particles,
            initial_field=initial_field,
            log_to_terminal=False,
        )
        with log.LoggerMuted():
            simulation_object.run()

        whole_far_field_object = ff.scattered_far_field(
            vacuum_wavelength=vacuum_wavelength,
            particle_list=object_particles,
            layer_system=layers,
        )

        whole_dscs_object = (
            np.sum(whole_far_field_object.azimuthal_integral(), axis=0) * np.pi / 180
        )

    return whole_dscs_surface, whole_dscs_object


if __name__ == "__main__":
    spheres_surface = Sphere_surface(
        number_of_cells=3, side_length=3.0, reflective_index=complex(2.0, 0.0)
    )

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

    spheres_surface.__spheres_add__(spheres_radius_list, coordinates_list)

    object = [
        smuthi.particles.Sphere(
            radius=0.1, position=[0, 0, 0], refractive_index=complex(5, 0.0)
        )
    ]

    loss = calculate_loss(spheres_surface, object, 0.5, [np.pi / 4, np.pi / 3])

    print(loss)
