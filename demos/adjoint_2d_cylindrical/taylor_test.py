from firedrake_adjoint import *
import numpy as np
from gadopt import *
ds_t = ds_t(degree=6)
dx = dx(degree=6)

newton_stokes_solver_parameters = {
    "snes_type": "newtonls",
    "snes_linesearch_type": "l2",
    "snes_max_it": 100,
    "snes_atol": 1e-10,
    "snes_rtol": 1e-5,
    "snes_stol": 0,
    # "snes_monitor": ':./newton.txt',
    "ksp_type": "preonly",
    "pc_tyoe": "lu",
    "pc_factor_mat_solver_type": "mumps",
    # "snes_view": none,
    "snes_converged_reason": None,
    "fiedsplit_0": {
        "ksp_converged_reason": None,
    },
    "fiedsplit_1": {
        "ksp_converged_reason": None,
    }}


def main():
    for case in ["damping", "smoothing", "Tobs", "uobs"]:
        try:
            all_taylor_tests(case)
        except Exception:
            raise Exception(f"Taylor test for case {case} failed!")


def all_taylor_tests(case):
    tape = get_working_tape()
    tape.clear_tape()

    # Set up geometry:
    rmax = 2.22
    rmax_earth = 6370  # Radius of Earth [km]
    rmin_earth = rmax_earth - 2900  # Radius of CMB [km]
    r_410_earth = rmax_earth - 410  # 410 radius [km]
    r_660_earth = rmax_earth - 660  # 660 raidus [km]
    r_410 = rmax - (rmax_earth - r_410_earth)/(rmax_earth - rmin_earth)
    r_660 = rmax - (rmax_earth - r_660_earth)/(rmax_earth - rmin_earth)

    with CheckpointFile('Checkpoint230.h5', mode='r') as chckpoint:
        mesh = chckpoint.load_mesh("firedrake_default_extruded")

    # enable_disk_checkpointint()
    # Set up function spaces - currently using the bilinear Q2Q1 element pair:
    V = VectorFunctionSpace(mesh, "CG", 2)  # Velocity function space (vector)
    W = FunctionSpace(mesh, "CG", 1)  # Pressure function space (scalar)
    Q = FunctionSpace(mesh, "CG", 2)  # Temperature function space (scalar)
    Q1 = FunctionSpace(mesh, "CG", 1)  # Control function space (scalar)
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
    Tic = Function(Q1, name="Initial Temperature")

    checkpoint_file = CheckpointFile("Checkpoint_State.h5", "r")
    # Initialise the control
    Tic.project(checkpoint_file.load_function(
        mesh,
        "Temperature",
        idx=max_timesteps-1))
    Taverage = checkpoint_file.load_function(
        mesh,
        "Average Temperature",
        idx=0)

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

    # Nullspaces and near-nullspaces:
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
    control = Control(Tic)

    u_misfit = 0

    # We need to project the initial condition from Q1 to Q2,
    # and impose the boundary conditions at the same time
    T.project(Tic, bcs=energy_solver.strong_bcs)

    # If it is only for smoothing or damping, there is no need to do the time-steping
    initial_timestep = 0 if case in ["Tobs", "uobs"] else max_timesteps

    # Now perform the time loop:
    for timestep in range(initial_timestep, max_timesteps):
        stokes_solver.solve()
        energy_solver.solve()
        # Load the velocity
        uobs = checkpoint_file.load_function(
            mesh,
            name="Velocity",
            idx=timestep
        )
        u_misfit += assemble(dot(u - uobs, u - uobs) * ds_t)

    # Load the observed final state
    Tobs = checkpoint_file.load_function(mesh, "Temperature", idx=max_timesteps - 1)
    Tobs.rename("Observed Temperature")

    # Load the reference initial state
    # Needed to measure performance of weightings
    Tic_ref = checkpoint_file.load_function(mesh, "Temperature", idx=0)
    Tic_ref.rename("Reference Initial Temperature")

    # Load the average temperature profile
    Taverage = checkpoint_file.load_function(mesh, "Average Temperature", idx=0)

    checkpoint_file.close()

    norm_obs = assemble(Tobs ** 2 * dx)

    if case == "smoothing":
        norm_smoothing = assemble(
            dot(grad(Taverage), grad(Taverage)) * dx)
        objective = (
            norm_obs * assemble(
                dot(grad(Tic-Taverage), grad(Tic-Taverage)) * dx) /
            norm_smoothing
        )
    elif case == "damping":
        norm_damping = assemble(
            0.5*(Taverage)**2 * dx)
        objective = norm_obs * assemble((Tic - Taverage)**2 * dx) / norm_damping
    elif case == "Tobs":
        objective = assemble((T - Tobs)**2 * dx)
    else:
        norm_u_surface = assemble(
            dot(uobs, uobs) * ds_t)
        objective = norm_obs * u_misfit / (max_timesteps) / norm_u_surface

    # Do not annotate from here on
    pause_annotation()

    # Defining the object for pyadjoint
    reduced_functional = ReducedFunctional(
        objective,
        control)

    Delta_temp = Function(Tic.function_space(), name="Delta_Temperature")
    Delta_temp.dat.data[:] = np.random.random(Delta_temp.dat.data.shape)
    minconv = taylor_test(reduced_functional, Tic, Delta_temp)

    log(
        (
            "\n\nEnd of Taylor Test ****: "
            f"case: {case}"
            f"conversion: {minconv:.8e}\n\n\n"
        )
    )
    # This is to make sure we are annotating
    continue_annotation()

    assert minconv > 1.9


if __name__ == "__main__":
    main()
