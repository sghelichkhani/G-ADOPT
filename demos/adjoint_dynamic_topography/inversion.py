from gadopt import *
from gadopt.inverse import *

left_id, right_id, bottom_id, top_id = 1, 2, 3, 4  # Boundary IDs

with CheckpointFile("checkpoint.h5", mode="r") as fi:
    mesh = fi.load_mesh(name="firedrake_default")
    #
    residual_topography = fi.load_function(mesh, f"force_{top_id}")


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
mu_control = Function(W, name="control").assign(0.0)
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
                             )

surface_force_solver = BoundaryNormalStressSolver(stokes_solver, top_id)

# Solve Stokes sytem:
stokes_solver.solve()

# compute `model` dynamic topography
surface_force = surface_force_solver.solve()

# form the objective function, between model and `data`
objective = assemble(0.5 * (surface_force - residual_topography) ** 2 * ds(top_id))

# Defining the reduced functional
reduced_functional = ReducedFunctional(objective, controls=control)


# Callback function for writing out the solution's visualisation
solution_pvd = VTKFile("solutions.pvd")


def callback():
    solution_pvd.write(mu_control.block_variable.checkpoint)


# Perform a bounded nonlinear optimisation where temperature
# is only permitted to lie in the range [0, 1]
mu_lb = Function(mu_control.function_space(), name="Lower bound temperature")
mu_ub = Function(mu_control.function_space(), name="Upper bound temperature")
mu_lb.assign(-2.0)
mu_ub.assign(2.0)

minimisation_problem = MinimizationProblem(reduced_functional, bounds=(mu_lb, mu_ub))

# Adjust minimisation parameters
minimisation_parameters["Status Test"]["Iteration Limit"] = 2

optimiser = LinMoreOptimiser(
    minimisation_problem,
    minimisation_parameters,
)
optimiser.add_callback(callback)
optimiser.run()


optimiser.rol_solver.rolvector.dat[0].rename("Final Solution")
with CheckpointFile("final_solution.h5", mode="w") as fi:
    fi.save_mesh(mesh)
    fi.save_function(optimiser.rol_solver.rolvector.dat[0])

VTKFile("final_solution.pvd").write(optimiser.rol_solver.rolvector.dat[0])
