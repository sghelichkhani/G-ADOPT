from gadopt import *

nx, ny = 40, 40  # Number of cells in x and y directions.
mesh = UnitSquareMesh(nx, ny, quadrilateral=True)  # Square mesh generated via firedrake
left_id, right_id, bottom_id, top_id = 1, 2, 3, 4  # Boundary IDs

V = VectorFunctionSpace(mesh, "CG", 2)  # Velocity function space (vector)
W = FunctionSpace(mesh, "CG", 1)  # Pressure function space (scalar)
Q = FunctionSpace(mesh, "CG", 2)  # Temperature function space (scalar)

Z = MixedFunctionSpace([V, W])  # Mixed function space.


z = Function(Z)  # A field over the mixed function space Z.
u, p = split(z)  # Returns symbolic UFL expression for u and p
z.subfunctions[0].rename("Velocity")
z.subfunctions[1].rename("Pressure")

# +
Ra = Constant(1e4)  # Rayleigh number
mu = Function(W, name="mu")
approximation = BoussinesqApproximation(Ra)

time = 0.0  # Initial time
delta_t = Constant(1e-6)  # Initial time-step

steady_state_tolerance = 1e-9  # Used to determine if solution has reached a steady state.

X = SpatialCoordinate(mesh)
T = Function(Q, name="Temperature")
T.interpolate((1.0-X[1]) + (0.05*cos(pi*X[0])*sin(pi*X[1])))
mu.interpolate(10 ** (-2*exp(-(X[1] - 0.8)**2/0.005)))
Z_nullspace = create_stokes_nullspace(Z, closed=True, rotational=False)

# +
stokes_bcs = {
    bottom_id: {'uy': 0},
    top_id: {'uy': 0},
    left_id: {'ux': 0},
    right_id: {'ux': 0},
}

temp_bcs = {
    bottom_id: {'T': 1.0},
    top_id: {'T': 0.0},
}

# +
stokes_solver = StokesSolver(z, T, approximation, bcs=stokes_bcs, mu=mu,
                             nullspace=Z_nullspace, transpose_nullspace=Z_nullspace,
                             cartesian=True, constant_jacobian=True)

surface_force_solver = BoundaryNormalStressSolver(stokes_solver, top_id)

# Solve Stokes sytem:
stokes_solver.solve()

# Write output:
surface_force = surface_force_solver.solve()

VTKFile("forward-visualisation.pvd").write(*z.subfunctions, T, surface_force, mu)

with CheckpointFile("checkpoint.h5", mode="w") as fi:
    fi.save_mesh(mesh)
    fi.save_function(surface_force)
