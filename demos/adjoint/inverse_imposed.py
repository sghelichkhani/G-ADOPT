from gadopt import *
from firedrake_adjoint import *

dx = dx(degree=6)

with CheckpointFile("mesh.h5", "r") as f:
    mesh = f.load_mesh("firedrake_default_extruded")

bottom_id, top_id = "bottom", "top"
left_id, right_id = 1, 2

domain_volume = assemble(1*dx(domain=mesh))

# Set up function spaces for the P2P1 pair
V = VectorFunctionSpace(mesh, "CG", 2)  # Velocity function space (vector)
W = FunctionSpace(mesh, "CG", 1)  # Pressure function space (scalar)
Q = FunctionSpace(mesh, "CG", 2)  # Temperature function space (scalar)
Q1 = FunctionSpace(mesh, "CG", 1)  # Average temperature function space (scalar, P1)
Z = MixedFunctionSpace([V, W])

q = TestFunction(Q)
q1 = TestFunction(Q1)

z = Function(Z)  # A field over the mixed function space Z
u, p = z.subfunctions  # Symbolic UFL expressions for u and p
u.rename("Velocity")
p.rename("Pressure")

T_ic = Function(Q1, name="Initial Temperature")
with CheckpointFile("final_state.h5", "r") as f:
    T_ic.project(f.load_function(mesh, "Temperature"))

T = Function(Q, name="Temperature")

u_ref = Function(V, name="Reference Velocity")
u_ref.assign(Constant(0.0))

Ra = Constant(1e6)
approximation = BoussinesqApproximation(Ra)

delta_t = Constant(4e-6)  # Constant time step
max_timesteps = 80
time = 0.0

Z_nullspace = create_stokes_nullspace(Z)

# Imposed velocity boundary condition on top, free-slip on other sides
stokes_bcs = {
    bottom_id: {"uy": 0},
    top_id: {"u": u_ref},
    left_id: {"ux": 0},
    right_id: {"ux": 0},
}
temp_bcs = {
    top_id: {"T": 0.0},
    bottom_id: {"T": 1.0},
}

solver_parameters = {
    "snes_type": "ksponly",
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
    "mat_type": "aij",
}

energy_solver = EnergySolver(
    T,
    u,
    approximation,
    delta_t,
    ImplicitMidpoint,
    bcs=temp_bcs,
    solver_parameters=solver_parameters
)
Told = energy_solver.T_old
Ttheta = 0.5*T + 0.5*Told
Told.assign(T)

stokes_solver = StokesSolver(
    z,
    Ttheta,
    approximation,
    bcs=stokes_bcs,
    nullspace=Z_nullspace,
    transpose_nullspace=Z_nullspace,
    solver_parameters=solver_parameters,
)

# Project the initial condition from Q1 to Q
ic_projection_problem = LinearVariationalProblem(
    q * TrialFunction(Q) * dx,
    q * T_ic * dx,
    T,
    bcs=energy_solver.strong_bcs,
)
ic_projection_solver = LinearVariationalSolver(ic_projection_problem)

ic_projection_solver.solve()

output_file = File("visualisation/output_inverse_imposed.pvd")
dump_period = 10

# Control variable for optimisation
control = Control(T_ic)

u_checkpoint = CheckpointFile("reference_velocity.h5", "r")

# Populate the tape by running the forward simulation
for timestep in range(0, max_timesteps):
    stokes_solver.solve()
    energy_solver.solve()
    time += float(delta_t)

    average_temperature = assemble(T * dx) / domain_volume
    log(f"{timestep} {time:.02e} {average_temperature:.1e}")

    if timestep % dump_period == 0:
        output_file.write(u, p, T)

    # load the reference velocity for the next timestep
    u_ref.assign(u_checkpoint.load_function(mesh, name="Velocity", idx=timestep))

with CheckpointFile("initial_state.h5", "r") as f:
    T_average = f.load_function(mesh, "Average Temperature")

with CheckpointFile("final_state.h5", "r") as f:
    T_reference = f.load_function(mesh, "Temperature")
    T_reference.rename("Reference Temperature")

alpha_smoothing = 1e-1  # [1e-1, 1e-2, 1e-3]
alpha_damping = 1e-2  # [1e-2, 1e-3, 1e-4]

X = SpatialCoordinate(mesh)

t_misfit = assemble((T - T_reference) ** 2 * dx)
norm_reference = assemble(T_reference ** 2 * dx)
norm_gradient = assemble(dot(grad(T_reference), grad(T_reference)) * dx)
smoothing = assemble(dot(grad(T_ic - T_average), grad(T_ic - T_average)) * dx)
damping_mask = gaussian(X[1], 1.0, 0.1) + gaussian(X[1], 0.0, 0.1)
damping = assemble(damping_mask * (T_ic - T_average) ** 2 * dx)

# Define the reduced functional for evaluating the optimisation
objective = (
    t_misfit +
    (alpha_smoothing * norm_reference / norm_gradient) * smoothing +
    alpha_damping * damping
)

reduced_functional = ReducedFunctional(objective, control)
log(f"reduced functional: {reduced_functional(T_average)}")
