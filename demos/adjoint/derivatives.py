from gadopt import *
from gadopt.inverse import *


def main():
    for case in ["damping", "smoothing", "Tobs", "uobs"]:
        try:
            all_taylor_tests(case)
        except Exception:
            raise Exception(f"Taylor test for case {case} failed!")


def all_taylor_tests(case):

    # Make sure we start from a clean tape
    tape = get_working_tape()
    tape.clear_tape()

    with CheckpointFile("mesh.h5", "r") as f:
        mesh = f.load_mesh("firedrake_default_extruded")

    # to enable checkpointing to disk
    enable_disk_checkpointing(dirname="./test/")

    bottom_id, top_id = "bottom", "top"
    left_id, right_id = 1, 2

    # Set up function spaces for the Q2Q1 pair
    V = VectorFunctionSpace(mesh, "CG", 2)  # Velocity function space (vector)
    W = FunctionSpace(mesh, "CG", 1)  # Pressure function space (scalar)
    Q = FunctionSpace(mesh, "CG", 2)  # Temperature function space (scalar)
    Q1 = FunctionSpace(mesh, "CG", 1)  # Average temperature function space (scalar, P1)
    Z = MixedFunctionSpace([V, W])

    z = Function(Z)  # A field over the mixed function space Z
    u, p = split(z)  # Symbolic UFL expressions for u and p

    # Without a restart to continue from, our initial guess is the final state of the forward run
    # We need to project the state from Q2 into Q1
    Tic = Function(Q1, name="Initial Temperature")

    # Temperature function
    T = Function(Q, name="Temperature")

    Ra = Constant(1e6)
    approximation = BoussinesqApproximation(Ra)

    delta_t = Constant(4e-6)  # Constant time step
    max_timesteps = 80
    init_timestep = 0 if case in ["Tobs", "uobs"] else max_timesteps

    Z_nullspace = create_stokes_nullspace(Z, closed=True, rotational=False)

    # the initial guess for the control
    with CheckpointFile("Checkpoint_State.h5", "r") as f:
        Tic.project(f.load_function(mesh, "Temperature", idx=max_timesteps - 1))

    # Imposed velocity boundary condition on top, free-slip on other sides
    stokes_bcs = {
        bottom_id: {"uy": 0},
        top_id: {"uy": 0},
        left_id: {"ux": 0},
        right_id: {"ux": 0},
    }
    temp_bcs = {
        top_id: {"T": 0.0},
        bottom_id: {"T": 1.0},
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
    )

    # Control variable for optimisation
    control = Control(Tic)

    checkpoint_file = CheckpointFile("Checkpoint_State.h5", "r")

    u_misfit = 0.0

    # We need to project the initial condition from Q1 to Q2,
    # and impose the boundary conditions at the same time
    T.project(Tic, bcs=energy_solver.strong_bcs)

    # Populate the tape by running the forward simulation
    for timestep in range(init_timestep, max_timesteps):
        stokes_solver.solve()

        # load the reference velocity
        uobs = checkpoint_file.load_function(
            mesh,
            name="Velocity",
            idx=timestep)
        u_misfit += assemble(0.5 * (uobs - u)**2 * ds_t)
        energy_solver.solve()

    # Load the observed final state
    Tobs = checkpoint_file.load_function(mesh, "Temperature", idx=max_timesteps - 1)
    Tobs.rename("ObservedTemperature")

    # Load the average temperature profile
    Taverage = checkpoint_file.load_function(mesh, "Average Temperature", idx=0)
    Taverage.rename("AverageTemperature")

    checkpoint_file.close()

    if case == "smoothing":
        norm_grad_Taverage = assemble(
            0.5*dot(grad(Taverage), grad(Taverage)) * dx)
        objective = 0.5 * assemble(dot(grad(Tic-Taverage), grad(Tic-Taverage)) * dx) / norm_grad_Taverage
    elif case == "damping":
        norm_Tavereage = assemble(
            0.5*(Taverage)**2 * dx)
        objective = 0.5 * assemble((Tic - Taverage)**2 * dx) / norm_Tavereage
    elif case == "Tobs":
        norm_final_state = assemble(
            0.5*(Tobs)**2 * dx)
        objective = 0.5 * assemble((T - Tobs)**2 * dx) / norm_final_state
    else:
        norm_u_surface = assemble(
            0.5 * (uobs)**2 * ds_t)
        objective = u_misfit / (max_timesteps) / norm_u_surface

    # All done with the forward run, stop annotating anything else to the tape
    pause_annotation()

    reduced_functional = ReducedFunctional(objective, control)

    # computing and storing the derivative
    derivative = reduced_functional.derivative(
        options={'riesz_representation': "L2"}
    )
    derivative.rename("derivative")

    # writing out derivative
    fi = File(f"derivative-{case}/derivative.pvd")
    fi.write(derivative)

    # make sure we keep annotating after this
    continue_annotation()


if __name__ == "__main__":
    main()
