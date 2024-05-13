"""
This runs the optimisation portion of the adjoint test case. A forward run first sets up
the tape with the adjoint information, then a misfit functional is constructed to be
used as the goal condition for nonlinear optimisation using ROL.
"""
from pathlib import Path
from gadopt import *
from gadopt.inverse import *


ds_t = ds_t(degree=6)
dx = dx(degree=6)


def main():
    inverse(alpha_u=1e-1, wavelength=0.1)


def stepwise_inverse(smoothing_weights, number_of_iterations, gnorms):
    # smoothing_weights = smoothing_weights  # [1.0, 0.5, 0.3, 0.1, 0.01]
    # number_of_iterations = number_of_iterations  # [15, 15, 15, 30, 50]
    total_number_of_iterations = 0
    for lambda_, iter_num, input_gnorm in zip(smoothing_weights, number_of_iterations, gnorms):
        total_number_of_iterations += iter_num
        inverse(alpha_u=1e-2, wavelength=lambda_, iteration_numbers=iter_num, total_number_of_iterations=total_number_of_iterations, input_gnorm=input_gnorm)


def inverse(alpha_u, wavelength, iteration_numbers, total_number_of_iterations, input_gnorm):
    """
    Use adjoint-based optimisation to solve for the initial condition of the rectangular
    problem.

    Parameters:
        alpha_u: The coefficient of the velocity misfit term
        wavelength: wavelength with which we can filter
    """
    # Clear the tape of any previous operations to ensure
    # the adjoint reflects the forward problem we solve here
    tape = get_working_tape()
    tape.clear_tape()

    script_dir = Path(__file__).parent
    my_iteration = find_iteration(script_dir)
    checkpoint_path = script_dir / "Checkpoint_State.h5"

    if my_iteration == 0:
        mesh_path = script_dir / "mesh.h5"
        with CheckpointFile(mesh_path.as_posix(), "r") as f:
            mesh = f.load_mesh("firedrake_default_extruded")
    else:
        mesh_path = script_dir / f"solution_{my_iteration:03d}.h5"
        with CheckpointFile(mesh_path.as_posix(), "r") as f:
            mesh = f.load_mesh("firedrake_default_extruded")

    bottom_id, top_id, left_id, right_id = "bottom", "top", 1, 2

    enable_disk_checkpointing(cleanup=False)

    # Set up function spaces for the Q2Q1 pair
    V = VectorFunctionSpace(mesh, "CG", 2)  # Velocity function space (vector)
    W = FunctionSpace(mesh, "CG", 1)  # Pressure function space (scalar)
    Q = FunctionSpace(mesh, "CG", 2)  # Temperature function space (scalar)
    Q1 = FunctionSpace(mesh, "CG", 1)  # Control function space
    Z = MixedFunctionSpace([V, W])  # Mixed function space

    # Test functions and functions to hold solutions:
    z = Function(Z)  # A field over the mixed function space Z
    u, p = z.subfunctions
    u.rename("Velocity")
    p.rename("Pressure")

    Ra = Constant(1e6)  # Rayleigh number
    approximation = BoussinesqApproximation(Ra)

    # Define time stepping parameters:
    max_timesteps = 80
    delta_t = Constant(4e-6)  # Constant time step

    # Without a restart to continue from, our initial guess is the final state of the forward run
    # We need to project the state from Q2 into Q1
    Tic = Function(Q1, name="Initial Temperature")

    checkpoint_file = CheckpointFile(checkpoint_path.as_posix(), "r")

    if my_iteration == 0:
        Tic.project(checkpoint_file.load_function(mesh, "Temperature", idx=max_timesteps - 1))
    else:
        with CheckpointFile(mesh_path.as_posix(), mode="r") as fi:
            # Initialise the control
            Tic.project(
                checkpoint_file.load_function(mesh, f"solution_{my_iteration:03d}")
            )

    # Temperature function in Q2, where we solve the equations
    T = Function(Q, name="Temperature")

    Z_nullspace = create_stokes_nullspace(Z, closed=True, rotational=False)

    stokes_bcs = {
        bottom_id: {"uy": 0},
        top_id: {"uy": 0},
        left_id: {"ux": 0},
        right_id: {"ux": 0},
    }
    temp_bcs = {
        bottom_id: {"T": 1.0},
        top_id: {"T": 0.0},
    }

    energy_solver = EnergySolver(
        T,
        u,
        approximation,
        delta_t,
        ImplicitMidpoint,
        bcs=temp_bcs,
    )

    stokes_solver = StokesSolver(
        z,
        T,
        approximation,
        bcs=stokes_bcs,
        nullspace=Z_nullspace,
        transpose_nullspace=Z_nullspace,
        constant_jacobian=True,
    )

    # Control variable for optimisation
    control = Control(Tic)

    u_misfit = 0.0
    # We need to project the initial condition from Q1 to Q2,
    # and impose the boundary conditions at the same time
    # Additionaly, we want to impose some measure of smoothness
    # for our solution. We do so by operating a diffusion equation
    smoother = DiffusiveSmoothingSolver(function_space=Q, wavelength=wavelength, bcs=temp_bcs)
    T.assign(smoother.action(Tic))

    # Populate the tape by running the forward simulation
    for timestep in range(0, max_timesteps):
        stokes_solver.solve()
        energy_solver.solve()

        # Update the accumulated surface velocity misfit using the observed value
        uobs = checkpoint_file.load_function(mesh, name="Velocity", idx=timestep)
        u_misfit += assemble(dot(u - uobs, u - uobs) * ds_t)

    # Load the observed final state
    Tobs = checkpoint_file.load_function(mesh, "Temperature", idx=max_timesteps - 1)
    Tobs.rename("Observed Temperature")

    # Load the reference initial state
    # Needed to measure performance of weightings
    Tic_ref = checkpoint_file.load_function(mesh, "Temperature", idx=0)
    Tic_ref.rename("Reference Initial Temperature")

    checkpoint_file.close()

    # Define the component terms of the overall objective functional
    norm_obs = assemble(Tobs**2 * dx)
    norm_u_surface = assemble(dot(uobs, uobs) * ds_t)

    # Temperature misfit between solution and observation
    t_misfit = assemble((T - Tobs) ** 2 * dx)

    objective = (
        t_misfit +
        alpha_u * (norm_obs * u_misfit / max_timesteps / norm_u_surface)
    )

    # All done with the forward run, stop annotating anything else to the tape
    pause_annotation()

    # Defining the object for pyadjoint
    reduced_functional = ReducedFunctional(objective, control)

    class CallbackClass:
        def __init__(self, iteration):
            self.T_ic_rec = Function(Tic.function_space())
            self.T_ref_rec = Function(T.function_space())
            self.iteration = iteration

        def __call__(self):
            with CheckpointFile(f"solution_{self.iteration:03d}.h5", mode="w") as fi:
                fi.save_mesh(mesh)
                fi.save_function(Tic.block_variable.checkpoint.restore(), name=f"solution_{self.iteration:03d}")

            self.T_ic_rec.interpolate(Tic.block_variable.checkpoint.restore() - Tic_ref)
            initial_misfit = assemble(self.T_ic_rec ** 2 * dx)

            self.T_ref_rec.assign(T.block_variable.checkpoint.restore() - Tobs)
            final_misfit = assemble(self.T_ref_rec ** 2 * dx)
            log(f"Initial misfit; {initial_misfit}; final misfit: {final_misfit}")

            self.T_ic_rec.interpolate(smoother.action(self.T_ic_rec))
            self.T_ref_rec.interpolate(smoother.action(self.T_ref_rec))
            initial_misfit = assemble(self.T_ic_rec ** 2 * dx)
            final_misfit = assemble(self.T_ref_rec ** 2 * dx)
            log(f"Smooth Initial misfit; {initial_misfit}; Smooth final misfit: {final_misfit}")

    # Perform a bounded nonlinear optimisation where temperature
    # is only permitted to lie in the range [0, 1]
    T_lb = Function(Tic.function_space(), name="Lower bound temperature")
    T_ub = Function(Tic.function_space(), name="Upper bound temperature")
    T_lb.assign(0.0)
    T_ub.assign(1.0)

    minimisation_problem = MinimizationProblem(reduced_functional, bounds=(T_lb, T_ub))
    minimisation_parameters["Status Test"]["Iteration Limit"] = total_number_of_iterations
    minimisation_parameters["General"]["Secant"]["Maximum Storage"] = 10
    minimisation_parameters["Step"]["Trust Region"]["Initial Radius"] = 0.1
    minimisation_parameters["Status Test"]["Gradient Tolerance"] = input_gnorm

    optimiser = LinMoreOptimiser(
        minimisation_problem,
        minimisation_parameters,
        # checkpoint_dir="checkpoints",
        auto_checkpoint=False,
    )
    optimiser.add_callback(CallbackClass(iteration=my_iteration))
    # if total_number_of_iterations != iteration_numbers:
    #     # print(total_number_of_iterations - iteration_numbers)
    #     # optimiser.restore(total_number_of_iterations-iteration_numbers)
    #     optimiser.restore()
    optimiser.run()

    # If we're performing mulitple successive optimisations, we want
    # to ensure the annotations are switched back on for the next code
    # to use them
    continue_annotation()


def set_minimisation_parameters(restore_from, num_iterations):
    return minimisation_parameters


def find_iteration(main_path):
    main_path = Path(main_path)
    all_files = list(main_path.glob(pattern="solution*.h5"))

    if len(all_files) == 0:
        iteration = 0
    else:
        iteration = sorted([int(fi.as_posix().split("/")[-1].split("_")[-1].split(".")[0]) for fi in all_files])[-1]

    return iteration


if __name__ == "__main__":
    main()
