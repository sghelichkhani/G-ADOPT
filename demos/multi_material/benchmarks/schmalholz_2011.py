from dataclasses import dataclass
from typing import ClassVar, Tuple

import firedrake as fd
import initial_signed_distance as isd
import matplotlib.pyplot as plt
import numpy as np
from mpi4py import MPI
from scipy.io import loadmat

import gadopt as ga


@dataclass
class Mantle(ga.Material):
    def viscosity(self, *args, **kwargs):
        return 1e21


@dataclass
class Lithosphere(ga.Material):
    visc_coeff: ClassVar[float] = 4.75e11
    stress_exponent: ClassVar[float] = 4.0
    visc_bounds: ClassVar[Tuple[float, float]] = (1e21, 1e25)

    @classmethod
    def viscosity(cls, *args, **kwargs):
        strain_rate = fd.sym(fd.grad(kwargs["velocity"]))
        strain_rate_sec_inv = fd.sqrt(fd.inner(strain_rate, strain_rate) / 2 + 1e-99)

        return fd.min_value(
            fd.max_value(
                cls.visc_coeff * strain_rate_sec_inv ** (1 / cls.stress_exponent - 1),
                cls.visc_bounds[0],
            ),
            cls.visc_bounds[1],
        )


class Simulation:
    name = "Schmalholz_2011"

    # Degree of the function space on which the level-set function is defined.
    level_set_func_space_deg = 2

    # Mesh resolution should be sufficient to capture eventual small-scale dynamics
    # in the neighbourhood of material interfaces tracked by the level-set approach.
    # Insufficient mesh refinement can lead to unwanted motion of material interfaces.
    domain_dimensions = (1e6, 6.6e5)
    mesh_file = "benchmarks/schmalholz_2011.msh"

    # The following two lists must be ordered such that, unpacking from the end, each
    # pair of arguments enables initialising a level set whose 0-contour corresponds to
    # the entire interface between a given material and the remainder of the numerical
    # domain. By convention, the material thereby isolated occupies the positive side
    # of the signed-distance level set.
    isd_params = [None]
    initialise_signed_distance = [isd.isd_schmalholz]

    # Material ordering must follow the logic implemented in the above two lists. In
    # other words, the last material in the below list corresponds to the portion of
    # the numerical domain entirely isolated by the level set initialised using the
    # first pair of arguments (unpacking from the end) in the above two lists.
    # Consequently, the first material in the below list occupies the negative side of
    # the level set resulting from the last pair of arguments above.
    mantle = Mantle(density=3150)
    lithosphere = Lithosphere(density=3300)
    materials = [mantle, lithosphere]
    reference_material = mantle

    # Physical parameters
    Ra, g = 1, 9.81

    # Boundary conditions
    temp_bcs = None
    stokes_bcs = {
        1: {"ux": 0, "uy": 0},
        2: {"ux": 0, "uy": 0},
        3: {"uy": 0},
        4: {"uy": 0},
    }

    # Timestepping objects
    dt = 1e11
    subcycles = 1
    time_end = 25e6 * 365.25 * 8.64e4
    dump_period = 5e5 * 365.25 * 8.64e4

    # Diagnostic objects
    diag_fields = {"normalised_time": [], "slab_necking": []}
    lithosphere_thickness = 8e4
    slab_length = 2.5e5
    slab_width = 8e4
    characteristic_time = (
        4
        * Lithosphere.visc_coeff
        / (lithosphere.density - mantle.density)
        / g
        / slab_length
    ) ** Lithosphere.stress_exponent

    @classmethod
    def initialise_temperature(cls, temperature):
        pass

    @classmethod
    def diagnostics(cls, simu_time, variables):
        epsilon = float(variables["epsilon"])
        level_set = variables["level_set"][0]
        level_set_data = level_set.dat.data_ro_with_halos
        coords_data = (
            fd.Function(
                fd.VectorFunctionSpace(level_set.ufl_domain(), level_set.ufl_element())
            )
            .interpolate(fd.SpatialCoordinate(level_set))
            .dat.data_ro_with_halos
        )

        mask_ls_outside = (
            (coords_data[:, 0] <= cls.domain_dimensions[0] / 2)
            & (
                coords_data[:, 1]
                < cls.domain_dimensions[1] - cls.lithosphere_thickness - 2e4
            )
            & (
                coords_data[:, 1]
                > cls.domain_dimensions[1] - cls.lithosphere_thickness - cls.slab_length
            )
            & (level_set_data < 0.5)
        )
        mask_ls_inside = (
            (coords_data[:, 0] <= cls.domain_dimensions[0] / 2)
            & (
                coords_data[:, 1]
                < cls.domain_dimensions[1] - cls.lithosphere_thickness - 2e4
            )
            & (
                coords_data[:, 1]
                > cls.domain_dimensions[1] - cls.lithosphere_thickness - cls.slab_length
            )
            & (level_set_data >= 0.5)
        )
        if mask_ls_outside.any():
            ind_outside = coords_data[mask_ls_outside, 0].argmax()
            hor_coord_outside = coords_data[mask_ls_outside, 0][ind_outside]
            if not mask_ls_outside.all():
                ver_coord_outside = coords_data[mask_ls_outside, 1][ind_outside]
                mask_ver_coord = (
                    abs(coords_data[mask_ls_inside, 1] - ver_coord_outside) < 1e3
                )
                if mask_ver_coord.any():
                    ind_inside = coords_data[mask_ls_inside, 0][mask_ver_coord].argmin()
                    hor_coord_inside = coords_data[mask_ls_inside, 0][mask_ver_coord][
                        ind_inside
                    ]

                    ls_outside = max(
                        1e-6,
                        min(1 - 1e-6, level_set_data[mask_ls_outside][ind_outside]),
                    )
                    sdls_outside = epsilon * np.log(ls_outside / (1 - ls_outside))

                    ls_inside = max(
                        1e-6,
                        min(
                            1 - 1e-6,
                            level_set_data[mask_ls_inside][mask_ver_coord][ind_inside],
                        ),
                    )
                    sdls_inside = epsilon * np.log(ls_inside / (1 - ls_inside))

                    ls_dist = sdls_inside / (sdls_inside - sdls_outside)
                    hor_coord_interface = (
                        ls_dist * hor_coord_outside + (1 - ls_dist) * hor_coord_inside
                    )
                    min_width = cls.domain_dimensions[0] - 2 * hor_coord_interface
                else:
                    min_width = cls.domain_dimensions[0] - 2 * hor_coord_outside
            else:
                min_width = cls.domain_dimensions[0] - 2 * hor_coord_outside
        else:
            min_width = np.inf

        min_width_global = level_set.comm.allreduce(min_width, MPI.MIN)

        cls.diag_fields["normalised_time"].append(simu_time / cls.characteristic_time)
        cls.diag_fields["slab_necking"].append(min_width_global / cls.slab_width)

    @classmethod
    def save_and_plot(cls):
        if MPI.COMM_WORLD.rank == 0:
            np.savez(f"{cls.name.lower()}/output", diag_fields=cls.diag_fields)

            slab_necking_schmalholz = loadmat("data/DET_FREE_NEW_TOP_R100.mat")

            fig, ax = plt.subplots(1, 1, figsize=(12, 10), constrained_layout=True)

            ax.grid()

            ax.set_xlabel("Normalised time")
            ax.set_ylabel("Slab necking")

            ax.plot(
                slab_necking_schmalholz["Time"][0] / cls.characteristic_time,
                slab_necking_schmalholz["Thickness"][0] / cls.slab_width * 1e3,
                label="Schmalholz (2011)",
            )

            ax.plot(
                cls.diag_fields["normalised_time"],
                cls.diag_fields["slab_necking"],
                label="Conservative level set",
            )

            ax.legend()

            fig.savefig(
                f"{cls.name}/slab_necking.pdf".lower(), dpi=300, bbox_inches="tight"
            )
