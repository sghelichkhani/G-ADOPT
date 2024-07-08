import numpy as np
from gadopt import *
from gadopt.inverse import *

left_id, right_id, bottom_id, top_id = 1, 2, 3, 4  # Boundary IDs

with CheckpointFile("checkpoint.h5", mode="r") as fi:
    mesh = fi.load_mesh(name="firedrake_default")
    residual_topography = fi.load_function(mesh, f"force_{top_id}")


V = VectorFunctionSpace(mesh, "CG", 2)  # Velocity function space (vector)
W = FunctionSpace(mesh, "CG", 1)  # Pressure function space (scalar)
Q = FunctionSpace(mesh, "CG", 2)  # Temperature function space (scalar)

Z = MixedFunctionSpace([V, W])  # Mixed function space.

z = Function(Z)  # A field over the mixed function space Z.
u, p = split(z)  # Returns symbolic UFL expression for u and p
z.subfunctions[0].rename("Velocity")
z.subfunctions[1].rename("Pressure")

Ra = Constant(1e4)  # Rayleigh number
mu_control = Function(W, name="control log viscosity").assign(0.0)
control = Control(mu_control)

kx = 1.0  # Conductivity in the x direction
ky = 0.0  # Conductivity in the y direction (set to zero for no diffusion)

# Construct the anisotropic conductivity tensor
# This tensor will have non-zero values only for the x component
ex = as_vector((1, 0))  # Unit vector in the x direction
ey = as_vector((0, 1))  # Unit vector in the y direction

K = kx * outer(ex, ex) + ky * outer(ey, ey)

smoother = DiffusiveSmoothingSolver(function_space=W, wavelength=1.0, K=K)
mu = Function(W, name="viscosity")
mu.project(10 ** smoother.action(mu_control))
approximation = BoussinesqApproximation(Ra)

X = SpatialCoordinate(mesh)
T = Function(Q, name="Temperature")

# We stick to this form of the temperature field for now
T.interpolate((1.0-X[1]) + (0.05*cos(pi*X[0])*sin(pi*X[1])))

Z_nullspace = create_stokes_nullspace(Z, closed=True, rotational=False)

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

stokes_solver = StokesSolver(z, T, approximation, bcs=stokes_bcs, mu=mu,
                             nullspace=Z_nullspace, transpose_nullspace=Z_nullspace,
                             cartesian=True)

surface_force_solver = BoundaryNormalStressSolver(stokes_solver, top_id)

# Solve Stokes sytem:
stokes_solver.solve()

# compute `model` dynamic topography
surface_force = surface_force_solver.solve()

# form the objective function, between model and `data`
objective = assemble(0.5 * (surface_force - residual_topography) ** 2 * ds(top_id))

# Defining the reduced functional
reduced_functional = ReducedFunctional(objective, controls=control)
der_func = reduced_functional.derivative(options={"riesz_representation": "L2"})
der_func.rename("derivative")

# Visualising the derivative
VTKFile("inverse-visualisation.pvd").write(*z.subfunctions, T, surface_force, der_func)

# Performing taylor test
Delta_mu = Function(mu.function_space(), name="Delta_Temperature")
Delta_mu.dat.data[:] = np.random.random(Delta_mu.dat.data.shape)

# Perform the Taylor test to verify the gradients
minconv = taylor_test(reduced_functional, mu, Delta_mu)
