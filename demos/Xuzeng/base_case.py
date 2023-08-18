from random import randint, uniform
from gadopt import *
import h5py
import numpy as np


def _main():
    for i in range(1000):
        run_fwd(i)


def initialiser(X):
    def gaussian(X_0, amp, sigma):
        return amp * exp(-((X - X_0) ** 2) / (2 * sigma**2))

    num_anomalies = randint(2, 8)
    anomaly = 0
    for i in range(num_anomalies):
        anomaly += gaussian(
            as_vector((uniform(0, 2.0), uniform(0, 1.0))),
            uniform(0.01, 0.2),
            uniform(0.02, 0.2),
        )
    return anomaly


class my_h5_file(object):
    def __init__(self, finame, description=""):
        with h5py.File(finame, mode="w") as f:
            f.attrs["description"] = description

        self.finame = finame
        self.counter = 0

    def write(self, T, intime):
        with h5py.File(self.finame, mode="a") as f:
            # Check if the datasets exist, and create or resize as necessary
            if "temperature" not in f:
                dataset = f.create_dataset(
                    "temperature", (0,) + T.shape, maxshape=(None,) + T.shape
                )
                timestamp = f.create_dataset(
                    "timestamps", (0,), maxshape=(None,), dtype="f"
                )
                timestamp.attrs["description"] = "Timestamps for the simulation"
                dataset.attrs["description"] = "Temperature field from a simulation"

            else:
                dataset = f["temperature"]
                timestamp = f["timestamps"]

            timestamp.resize((self.counter + 1,))
            dataset.resize((self.counter + 1,) + T.shape)
            dataset[self.counter] = T
            timestamp[self.counter] = intime
            self.counter += 1


def run_fwd(iteration_number):
    # defining a mesh to define average
    x_max = 2.0
    y_max = 1.0

    #  how many intervals along x directions
    disc_n = 100

    # 1D mesh
    mesh1d = IntervalMesh(disc_n * 2, length_or_left=0.0, right=x_max)

    # extruding mesh1d in y direction
    mesh = ExtrudedMesh(
        mesh=mesh1d,
        layers=disc_n,
        layer_height=y_max / disc_n,
        extrusion_type="uniform",
    )

    left_id, right_id, bottom_id, top_id = 1, 2, "bottom", "top"  # Boundary IDs

    # Set up function spaces - currently using the bilinear Q2Q1 element pair:
    V = VectorFunctionSpace(mesh, "CG", 2)  # Velocity function space (vector)
    W = FunctionSpace(mesh, "CG", 1)  # Pressure function space (scalar)
    Q = FunctionSpace(mesh, "CG", 2)  # Temperature function space (scalar)
    Z = MixedFunctionSpace([V, W])  # Mixed function space.

    # Function to store the solutions:
    z = Function(Z)  # a field over the mixed function space Z.
    u, p = split(z)  # Returns symbolic UFL expression for u and p

    # Set up temperature field and initialise:
    X = SpatialCoordinate(mesh)
    T = Function(Q, name="Temperature")
    T.interpolate(1.0 - X[1] + initialiser(X))

    delta_t = Constant(1e-6)  # Initial time-step
    t_adapt = TimestepAdaptor(delta_t, V, maximum_timestep=0.1, increase_tolerance=1.5)
    time = 0

    # Stokes related constants (note that since these are included in UFL, they are wrapped inside Constant):
    Ra = Constant(1e6)  # Rayleigh number
    approximation = BoussinesqApproximation(Ra)

    max_timesteps = 100

    # Nullspaces and near-nullspaces:
    Z_nullspace = create_stokes_nullspace(Z, closed=True, rotational=False)

    # Write output files in VTK format:
    u, p = z.subfunctions
    # Next rename for output:
    u.rename("Velocity")
    p.rename("Pressure")

    temp_bcs = {
        bottom_id: {"T": 1.0},
        top_id: {"T": 0.0},
    }

    stokes_bcs = {
        bottom_id: {"uy": 0},
        top_id: {"uy": 0},
        left_id: {"ux": 0},
        right_id: {"ux": 0},
    }

    energy_solver = EnergySolver(
        T, u, approximation, delta_t, ImplicitMidpoint, bcs=temp_bcs
    )
    stokes_solver = StokesSolver(
        z,
        T,
        approximation,
        bcs=stokes_bcs,
        cartesian=True,
        nullspace=Z_nullspace,
        transpose_nullspace=Z_nullspace,
    )

    output_file = my_h5_file(f"solution_{iteration_number}.h5")

    # Now perform the time loop:
    for timestep in range(0, max_timesteps):
        # Solve Stokes sytem:
        stokes_solver.solve()
        dt = t_adapt.update_timestep(u)
        # Temperature system:
        energy_solver.solve()
        time += dt
        output_file.write(
            np.array([T.dat.data[i :: disc_n * 2 + 1] for i in range(disc_n * 2 + 1)]),
            timestep,
        )
        # Calculate L2-norm of change in temperature:
        maxchange = sqrt(assemble((T - energy_solver.T_old) ** 2 * dx))
        log(maxchange)

        # Leave if steady-state has been achieved:
        if maxchange < 1e-3 and timestep > 100:
            break


if __name__ == "__main__":
    _main()
