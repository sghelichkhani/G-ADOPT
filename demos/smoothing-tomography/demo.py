"""
Isotropic and Anisotropic Smoothing Demonstration
=================================================

This script demonstrates the application of isotropic and anisotropic diffusive smoothing techniques on a cylindrical temperature field using the gadopt platform. The purpose is to illustrate how different diffusion properties can affect the smoothing behavior.

"""

from gadopt import *
from pathlib import Path

# input_finame = Path(__file__).parents[1] / "adjoint_2d_cylindrical/Checkpoint230.h5"
input_finame = Path(__file__).resolve().parent / "final_state.h5"

# Load a cylindrical temperature field from a checkpoint file
with CheckpointFile(str(input_finame), mode="r") as f:
    mesh = f.load_mesh("firedrake_default_extruded")
    T = f.load_function(mesh, "Temperature")

# Define boundary conditions for the temperature field
temp_bcs = {
    "bottom": {'T': 1.0},  # Fixed temperature at the bottom
    "top": {'T': 0.0},     # Fixed temperature at the top
}

# Compute layer average of the temperature for initial comparison
# This helps to visualize changes pre and post smoothing
T_avg = Function(T.function_space(), name='Layer_Averaged_Temp')
T_dev = Function(T.function_space(), name='Deviatoric_Smooth_Temp')
averager = LayerAveraging(mesh, cartesian=False, quad_degree=6)
averager.extrapolate_layer_average(T_avg, averager.get_layer_average(T))

# If K is going to be a function of temperature
K = 1.0 / (1 + exp(-20 * (T - T_avg)))

# Isotropic Smoothing
# -------------------
# In isotropic smoothing, we assume that the diffusion coefficient is the same in all directions.
# This simplifies the diffusion tensor to a scalar value, promoting uniform smoothing across all spatial directions.
smooth_solution = Function(T.function_space(), name="Smooth_Temperature")
smoother_isotropic = DiffusiveSmoothingSolver(
    function_space=T.function_space(),
    wavelength=0.1,  # Smoothing duration
    bcs=temp_bcs,
    K=K)

smooth_solution.assign(smoother_isotropic.action(T))
T_dev.assign(smooth_solution - T_avg)

with CheckpointFile("smoothed_input.h5", mode="w") as fi:
    fi.save_mesh(mesh)
    fi.save_function(T, name="TemperatureUnfiltered")
    fi.save_function(smooth_solution, name="TemperatureFiltered")
    fi.save_function(T_dev, name="TDeviatoric")


VTKFile("isotropic_smoothing.pvd").write(smooth_solution, T_dev, T_avg, T)
