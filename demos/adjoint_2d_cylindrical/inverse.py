from gadopt import *
import numpy as np

ds_t = ds_t(degree=6)
dx = dx(degree=6)

newton_stokes_solver_parameters = {
    "snes_type": "newtonls",
    "snes_linesearch_type": "l2",
    "snes_max_it": 100,
    "snes_atol": 1e-10,
    "snes_rtol": 1e-5,
    "snes_stol": 0,
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
    "snes_converged_reason": None,
    "fieldsplit_0": {
        "ksp_converged_reason": None,
    },
    "fieldsplit_1": {
        "ksp_converged_reason": None,
    },
}


def main():
    inverse(alpha_u=1e-1, alpha_d=1e-2, alpha_s=1e-1)


def inverse(alpha_u, alpha_d, alpha_s):
    """
    Use adjoint-based optimisation to solve for the initial condition of the cylindrical
    problem.

    Parameters:
        alpha_u: The coefficient of the velocity misfit term
        alpha_d: The coefficient of the initial condition damping term
        alpha_s: The coefficient of the smoothing term
    """

    # Clear the tape of any previous operations to ensure
    # the adjoint reflects the forward problem we solve here
    tape = get_working_tape()
    tape.clear_tape()

    # Set up geometry:
    rmin, rmax, nlayers = 1.22, 2.22, 128
    rmax_earth = 6370  # Radius of Earth [km]
    rmin_earth = rmax_earth - 2900  # Radius of CMB [km]
    r_410_earth = rmax_earth - 410  # 410 radius [km]
    r_660_earth = rmax_earth - 660  # 660 raidus [km]
    r_410 = rmax - (rmax_earth - r_410_earth)/(rmax_earth - rmin_earth)
    r_660 = rmax - (rmax_earth - r_660_earth)/(rmax_earth - rmin_earth)

    # Start with a previously-initialised temperature field
    with CheckpointFile("Checkpoint_State.h5", mode="r") as f:
        mesh = f.load_mesh("firedrake_default_extruded")

    enable_disk_checkpointing()

    # Set up function spaces - currently using the bilinear Q2Q1 element pair:
    V = VectorFunctionSpace(mesh, "CG", 2)  # Velocity function space (vector)
    W = FunctionSpace(mesh, "CG", 1)  # Pressure function space (scalar)
    Q = FunctionSpace(mesh, "CG", 2)  # Temperature function space (scalar)
    Q1 = FunctionSpace(mesh, "CG", 1)  # Temperature average (scalar)
    Z = MixedFunctionSpace([V, W])  # Mixed function space.

    # Test functions and functions to hold solutions:
    z = Function(Z)  # a field over the mixed function space Z.
    u, p = split(z)  # Returns symbolic UFL expression for u and p

    X = SpatialCoordinate(mesh)
    r = sqrt(X[0]**2 + X[1]**2)
    Ra = Constant(1e7)  # Rayleigh number
    approximation = BoussinesqApproximation(Ra)

    # Define time stepping parameters:
    max_timesteps = 200
    delta_t = Constant(5e-6)  # Initial time-step

    # Without a restart to continue from, our initial guess is the final state of the forward run
    # We need to project the state from Q2 into Q1
    T_ic = Function(Q1, name="Initial Temperature")
    with CheckpointFile("Checkpoint_State.h5", "r") as f:
        T_ic.project(f.load_function(mesh, "Temperature", idx=max_timesteps - 1))

    # Temperature function in Q2, where we solve the equations
    T = Function(Q, name="Temperature")

    # A step function designed to design viscosity jumps
    # Build a step centred at "centre" with given magnitude
    # Increase with radius if "increasing" is True
    def step_func(centre, mag, increasing=True, sharpness=50):
        return mag * (0.5 * (1 + tanh((1 if increasing else -1)*(r-centre)*sharpness)))

    # From this point, we define a depth-dependent viscosity mu
    mu_lin = 2.0

    # Assemble the depth dependence
    for line, step in zip(
            [5.*(rmax-r), 1., 1.],
            [step_func(r_660, 30, False),
             step_func(r_410, 10, False),
             step_func(2.2, 10, True)]):
        mu_lin += line*step

    # Add temperature dependence of viscosity
    mu_lin *= exp(-ln(Constant(80)) * T)

    # Assemble the viscosity expression in terms of velocity u
    eps = sym(grad(u))
    epsii = sqrt(0.5*inner(eps, eps))
    sigma_y = 1e4 + 2.0e5*(rmax-r)
    mu_plast = 0.1 + (sigma_y / epsii)
    mu_eff = 2 * (mu_lin * mu_plast)/(mu_lin + mu_plast)
    mu = conditional(mu_eff > 0.4, mu_eff, 0.4)

    # Radial average temperature function
    Taverage = Function(Q1, name="AverageTemperature")

    # Calculate the layer average of the initial state
    averager = LayerAveraging(
        mesh,
        np.linspace(rmin, rmax, nlayers*2),
        cartesian=False,
        quad_degree=6
    )
    averager.extrapolate_layer_average(
        Taverage,
        averager.get_layer_average(T)
    )

    Z_nullspace = create_stokes_nullspace(Z, closed=True, rotational=True)
    Z_near_nullspace = create_stokes_nullspace(Z, closed=False, rotational=True, translations=[0, 1])

    temp_bcs = {
        "bottom": {"T": 1.0},
        "top": {"T": 0.0},
    }
    stokes_bcs = {
        "bottom": {"un": 0},
        "top": {"un": 0},
    }

    energy_solver = EnergySolver(T, u, approximation, delta_t, ImplicitMidpoint, bcs=temp_bcs)
    stokes_solver = StokesSolver(
        z,
        T,
        approximation,
        mu=mu,
        bcs=stokes_bcs,
        cartesian=False,
        nullspace=Z_nullspace,
        transpose_nullspace=Z_nullspace,
        near_nullspace=Z_near_nullspace,
        solver_parameters=newton_stokes_solver_parameters
    )

    # Control variable for optimisation
    control = Control(T_ic)

    checkpoint_file = CheckpointFile("Checkpoint_State.h5", "r")
    u_misfit = 0

    # We need to project the initial condition from Q1 to Q2,
    # and impose the boundary conditions at the same time
    T.project(T_ic, bcs=energy_solver.strong_bcs)

    # Now perform the time loop:
    for timestep in range(0, max_timesteps):
        stokes_solver.solve()
        energy_solver.solve()

        # Update the accumulated surface velocity misfit using the observed value
        u_obs = checkpoint_file.load_function(
            mesh,
            name="Velocity",
            idx=timestep
        )
        u_misfit += assemble(dot(u - u_obs, u - u_obs) * ds_t)

    # Load the observed final state
    T_obs = checkpoint_file.load_function(mesh, "Temperature", idx=max_timesteps - 1)
    T_obs.rename("Observed Temperature")

    # Load the reference initial state
    # Needed to measure performance of weightings
    T_ic_ref = checkpoint_file.load_function(mesh, "Temperature", idx=0)
    T_ic_ref.rename("Reference Initial Temperature")

    # Load the average temperature profile
    T_average = checkpoint_file.load_function(mesh, "Average Temperature", idx=0)

    checkpoint_file.close()

    # Define the component terms of the overall objective functional
    damping = assemble((T_ic - T_average) ** 2 * dx)
    norm_damping = assemble(T_average ** 2 * dx)
    smoothing = assemble(dot(grad(T_ic - T_average), grad(T_ic - T_average)) * dx)
    norm_smoothing = assemble(dot(grad(T_obs), grad(T_obs)) * dx)
    norm_obs = assemble(T_obs ** 2 * dx)
    norm_u_surface = assemble(dot(u_obs, u_obs) * ds_t)

    # Temperature misfit between solution and observation
    t_misfit = assemble((T - T_obs) ** 2 * dx)

    objective = (
        t_misfit +
        alpha_u * (norm_obs * u_misfit / max_timesteps / norm_u_surface) +
        alpha_d * (norm_obs * damping / norm_damping) +
        alpha_s * (norm_obs * smoothing / norm_smoothing)
    )

    # All done with the forward run, stop annotating anything else to the tape
    pause_annotation()

    reduced_functional = ReducedFunctional(objective, control)

    def callback():
        initial_misfit = assemble((T_ic.block_variable.checkpoint.restore() - T_ic_ref) ** 2 * dx)
        final_misfit = assemble((T.block_variable.checkpoint.restore() - T_obs) ** 2 * dx)

        log(f"Initial misfit; {initial_misfit}; final misfit: {final_misfit}")

    # Perform a bounded nonlinear optimisation where temperature
    # is only permitted to lie in the range [0, 1]
    T_lb = Function(Q1, name="Lower bound temperature")
    T_ub = Function(Q1, name="Upper bound temperature")
    T_lb.assign(0.0)
    T_ub.assign(1.0)

    minimisation_problem = MinimizationProblem(reduced_functional, bounds=(T_lb, T_ub))

    optimiser = LinMoreOptimiser(
        minimisation_problem,
        minimisation_parameters,
        checkpoint_dir="optimisation_checkpoint"
    )
    optimiser.add_callback(callback)
    optimiser.run()

    # If we're performing mulitple successive optimisations, we want
    # to ensure the annotations are switched back on for the next code
    # to use them
    continue_annotation()


if __name__ == "__main__":
    main()
