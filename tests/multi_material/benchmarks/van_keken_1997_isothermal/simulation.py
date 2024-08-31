"""Compositional benchmark.
Van Keken, P. E., King, S. D., Schmeling, H., Christensen, U. R., Neumeister, D.,
& Doin, M. P. (1997).
A comparison of methods for the modeling of thermochemical convection.
Journal of Geophysical Research: Solid Earth, 102(B10), 22477-22495.
"""

from functools import partial

import initial_signed_distance as isd
import matplotlib.pyplot as plt
import numpy as np
from mpi4py import MPI

import gadopt as ga

from .materials import buoyant_material, dense_material


def diagnostics(simu_time, geo_diag, diag_vars, output_path):
    diag_fields["output_time"].append(simu_time)
    diag_fields["rms_velocity"].append(geo_diag.u_rms())
    diag_fields["entrainment"].append(
        ga.entrainment(
            diag_vars["level_set"][0],
            diag_params["domain_dim_x"] * diag_params["material_interface_y"],
            diag_params["entrainment_height"],
        )
    )

    if MPI.COMM_WORLD.rank == 0:
        np.savez(
            f"{output_path}/output_{checkpoint_restart}_check", diag_fields=diag_fields
        )


def plot_diagnostics(output_path):
    if MPI.COMM_WORLD.rank == 0:
        rms_vel_van_keken = np.loadtxt("data/pvk80_001.vrms.dat")
        entr_van_keken = np.loadtxt("data/pvk80_001.entr.dat")

        fig, ax = plt.subplots(1, 2, figsize=(18, 10), constrained_layout=True)

        ax[0].grid()
        ax[1].grid()

        ax[0].set_xlabel("Time (non-dimensional)")
        ax[1].set_xlabel("Time (non-dimensional)")
        ax[0].set_ylabel("Root-mean-square velocity (non-dimensional)")
        ax[1].set_ylabel("Entrainment (non-dimensional)")

        ax[0].plot(
            rms_vel_van_keken[:, 0],
            rms_vel_van_keken[:, 1],
            linestyle="dotted",
            label="PvK (van Keken et al., 1997)",
        )
        ax[1].plot(
            entr_van_keken[:, 0],
            entr_van_keken[:, 1],
            linestyle="dotted",
            label="PvK (van Keken et al., 1997)",
        )

        ax[0].plot(
            diag_fields["output_time"],
            diag_fields["rms_velocity"],
            label="Conservative level set",
        )
        ax[1].plot(
            diag_fields["output_time"],
            diag_fields["entrainment"],
            label="Conservative level set",
        )

        ax[0].legend(fontsize=12, fancybox=True, shadow=True)
        ax[1].legend(fontsize=12, fancybox=True, shadow=True)

        fig.savefig(
            f"{output_path}/rms_velocity_and_entrainment.pdf",
            dpi=300,
            bbox_inches="tight",
        )


# 0 indicates the initial run and positive integers corresponding restart runs.
checkpoint_restart = 0

# Mesh resolution should be sufficient to capture eventual small-scale dynamics
# in the neighbourhood of material interfaces tracked by the level-set approach.
# Insufficient mesh refinement can lead to unwanted motion of material interfaces.
domain_dims = (0.9142, 1)
mesh_gen = "firedrake"
mesh_elements = (128, 128)

# Parameters to initialise level sets
material_interface_y = 0.2
interface_deflection = 0.02
# The following two lists must be ordered such that, unpacking from the end, each
# pair of arguments enables initialising a level set whose 0-contour corresponds to
# the entire interface between a given material and the remainder of the numerical
# domain. By convention, the material thereby isolated occupies the positive side
# of the signed-distance level set.
initialise_signed_distance = [
    partial(isd.isd_simple_curve, domain_dims[0], isd.cosine_curve)
]
isd_params = [(interface_deflection, 2 * domain_dims[0], material_interface_y)]

# Material ordering must follow the logic implemented in the above two lists. In
# other words, the last material in the below list corresponds to the portion of
# the numerical domain entirely isolated by the level set initialised using the
# last pair of arguments in the above two lists. The first material in the below list
# must, therefore, occupy the negative side of the signed-distance level set initialised
# from the first pair of arguments above.
materials = [buoyant_material, dense_material]

# Approximation parameters
dimensional = False
buoyancy_terms = ["compositional"]

# Boundary conditions with mapping {1: left, 2: right, 3: bottom, 4: top}
stokes_bcs = {1: {"ux": 0}, 2: {"ux": 0}, 3: {"ux": 0, "uy": 0}, 4: {"ux": 0, "uy": 0}}

# Timestepping objects
initial_timestep = 1
dump_period = 10
checkpoint_period = 5
time_end = 2000

# Diagnostic objects
diag_fields = {"output_time": [], "rms_velocity": [], "entrainment": []}
entrainment_height = 0.2
diag_params = {
    "domain_dim_x": domain_dims[0],
    "material_interface_y": material_interface_y,
    "entrainment_height": entrainment_height,
}
