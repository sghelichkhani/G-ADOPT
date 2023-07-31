from gadopt import *
import numpy as np


def main():
    generate_mesh()
    run_forward()


def run_forward():

    with CheckpointFile("mesh.h5", mode="r") as fi:
        mesh = fi.load_mesh("firedrake_default_extruded")

    bottom_id, top_id = "bottom", "top"
    left_id, right_id = 1, 2

    # Set up function spaces for the Q2Q1 pair
    V = VectorFunctionSpace(mesh, "CG", 2)  # Velocity function space (vector)
    W = FunctionSpace(mesh, "CG", 1)  # Pressure function space (scalar)
    Q = FunctionSpace(mesh, "CG", 2)  # Temperature function space (scalar)
    Q1 = FunctionSpace(mesh, "CG", 1)  # Average temperature function space (scalar, P1)
    Z = MixedFunctionSpace([V, W])

    z = Function(Z)  # A field over the mixed function space Z
    u, p = z.subfunctions  # Symbolic UFL expressions for u and p
    u.rename("Velocity")
    p.rename("Pressure")

    T = Function(Q, name="Temperature")
    X = SpatialCoordinate(mesh)
    T.interpolate(
        0.5 * (erf((1 - X[1]) * 3.0) + erf(-X[1] * 3.0) + 1) +
        0.1 * exp(-0.5 * ((X - as_vector((0.5, 0.2))) / Constant(0.1)) ** 2)
    )

    T_average = Function(Q1, name="Average Temperature")

    # Calculate the layer average of the initial state
    averager = LayerAveraging(
        mesh,
        np.linspace(0, 1.0, 150*2),
        cartesian=True, quad_degree=6)
    averager.extrapolate_layer_average(
        T_average,
        averager.get_layer_average(T)
    )

    checkpoint_file = CheckpointFile("Checkpoint_State.h5", "w")
    checkpoint_file.save_mesh(mesh)
    checkpoint_file.save_function(T_average, idx=0)

    Ra = Constant(1e6)
    approximation = BoussinesqApproximation(Ra)

    delta_t = Constant(4e-6)  # Constant time step
    max_timesteps = 80
    time = 0.0

    Z_nullspace = create_stokes_nullspace(Z, closed=True, rotational=False)

    # Free-slip velocity boundary condition on all sides
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
        bcs=temp_bcs
    )

    stokes_solver = StokesSolver(
        z,
        T,
        approximation,
        bcs=stokes_bcs,
        nullspace=Z_nullspace,
        transpose_nullspace=Z_nullspace
    )

    output_file = File("vtu-files/output.pvd")
    dump_period = 10

    for timestep in range(0, max_timesteps):
        stokes_solver.solve()
        energy_solver.solve()
        time += float(delta_t)

        # Storing velocity to be used in the objective F
        checkpoint_file.save_function(u, idx=timestep)

        if timestep % dump_period == 0 or timestep == max_timesteps - 1:
            output_file.write(u, p, T)

    # Save the reference final temperature
    checkpoint_file.save_function(T, idx=max_timesteps - 1)
    checkpoint_file.close()


def generate_mesh():
    # Domain extents
    x_max = 1.0
    y_max = 1.0

    # Number of intervals along x direction
    disc_n = 150

    # Interval mesh in x direction, to be extruded along y
    mesh1d = IntervalMesh(disc_n, length_or_left=0.0, right=x_max)
    mesh_temp = ExtrudedMesh(
        mesh1d,
        layers=disc_n,
        layer_height=y_max / disc_n,
        extrusion_type="uniform"
    )

    # write out the mesh
    with CheckpointFile("mesh.h5", mode='w') as fi:
        fi.save_mesh(mesh_temp)


if __name__ == "__main__":
    main()
