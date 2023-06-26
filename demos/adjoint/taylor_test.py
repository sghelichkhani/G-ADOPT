from gadopt import *
from firedrake_adjoint import *
import numpy as np

dx = dx(degree=6)

with CheckpointFile("mesh.h5", "r") as f:
    mesh = f.load_mesh("firedrake_default_extruded")

bottom_id, top_id = "bottom", "top"
left_id, right_id = 1, 2

domain_volume = assemble(1*dx(domain=mesh))

# Set up function spaces for the Q2Q1 pair
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

# Without a restart to continue from, our initial guess is the final state of the forward run
# We need to project the state from Q2 into Q1
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

Z_nullspace = create_stokes_nullspace(Z, closed=True, rotational=False)

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
temp_bcs_q1 = [
    DirichletBC(Q1, 0.0, top_id),
    DirichletBC(Q1, 1.0, bottom_id),
]

energy_solver = EnergySolver(
    T,
    u,
    approximation,
    delta_t,
    ImplicitMidpoint,
    bcs=temp_bcs,
)
Told = energy_solver.T_old
Ttheta = 0.5*T + 0.5*Told

stokes_solver = StokesSolver(
    z,
    Ttheta,
    approximation,
    bcs=stokes_bcs,
    nullspace=Z_nullspace,
    transpose_nullspace=Z_nullspace,
)

# Define a simple problem to apply the imposed boundary condition to the IC
T_ = Function(Q1)
bc_problem = LinearVariationalProblem(
    q1 * TrialFunction(Q1) * dx,
    q1 * T_ * dx,
    T_ic,
    bcs=temp_bcs_q1,
)
bc_solver = LinearVariationalSolver(bc_problem)

# Project the initial condition from Q1 to Q
ic_projection_problem = LinearVariationalProblem(
    q * TrialFunction(Q) * dx,
    q * T_ic * dx,
    Told,
    bcs=energy_solver.strong_bcs,
)
ic_projection_solver = LinearVariationalSolver(ic_projection_problem)

# Control variable for optimisation
control = Control(T_ic)

# Apply the boundary condition to the control
# and obtain the initial condition
T_.assign(T_ic)
bc_solver.solve()
ic_projection_solver.solve()
T.assign(Told)

u_checkpoint = CheckpointFile("reference_velocity.h5", "r")

# Populate the tape by running the forward simulation
for timestep in range(0, max_timesteps):
    stokes_solver.solve()
    energy_solver.solve()
    time += float(delta_t)

    average_temperature = assemble(T * dx) / domain_volume
    log(f"{timestep} {time:.02e} {average_temperature:.1e}")

    # load the reference velocity for the next timestep since this is
    # saved after the solves in the forward run, we reload after the
    # solves here
    u_ref.assign(u_checkpoint.load_function(mesh, name="Velocity", idx=timestep))

with CheckpointFile("final_state.h5", "r") as f:
    T_reference = f.load_function(mesh, "Temperature")
    T_reference.rename("Reference Temperature")

t_misfit = assemble(0.5 * (T - T_reference) ** 2 * dx)
reduced_functional = ReducedFunctional(t_misfit, control)

# All done with the forward run, stop annotating anything else to the tape
pause_annotation()

delta_T = Function(Q1, name="Delta Temperature")
delta_T.dat.data[:] = np.random.random(delta_T.dat.data.shape)
log(taylor_test(reduced_functional, T_ic, delta_T))
