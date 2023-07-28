from gadopt import *
from mpi4py import MPI

# Set up geometry:
nx, ny = 40, 40
L = 3e6 # length of domain in m
D = L # Depth of the domain in m
mesh = RectangleMesh(nx, ny, L, D)  # Rectangle mesh generated via firedrake
left_id, right_id, bottom_id, top_id = 1, 2, 3, 4  # Boundary IDs

# Set up function spaces - currently using the bilinear Q2Q1 element pair:
V = VectorFunctionSpace(mesh, "CG", 2)  # Displacement function space (vector)
W = FunctionSpace(mesh, "CG", 1)  # Pressure function space (scalar)
Q = FunctionSpace(mesh, "CG", 2)  # Temperature function space (scalar)
Z = MixedFunctionSpace([V, W])  # Mixed function space.

# Function to store the solutions:
z = Function(Z)  # a field over the mixed function space Z.
u, p = split(z)  # Returns symbolic UFL expression for u and p

u_old = Function(V)
u_old.project(u)
displacement = Function(V).assign(0)

T = Function(Q, name="Temperature").assign(0)
# Output function space information:
log("Number of Velocity DOF:", V.dim())
log("Number of Pressure DOF:", W.dim())
log("Number of Velocity and Pressure DOF:", V.dim()+W.dim())
log("Number of Temperature DOF:", Q.dim())

# timestepping 

viscosity = Constant(1e21)  # Viscosity Pa s
shear_modulus = Constant(1e11) # Shear modulus in Pa
maxwell_time = viscosity / shear_modulus
dt = Constant(0.125*maxwell_time)  # Initial time-step
effective_viscosity = Constant(viscosity/(maxwell_time+0.5*dt))
prefactor_prestress = Constant((maxwell_time-0.5*dt)/(maxwell_time + 0.5*dt))

print("effective visc", effective_viscosity.values()[0])
print("prefactor prestress", prefactor_prestress.values()[0])

u_,p_ = z.split()
previous_stress =  prefactor_prestress * 2 * effective_viscosity * sym(grad(u_old))

time = 0.0
max_timesteps = 250
# Stokes related constants (note that since these are included in UFL, they are wrapped inside Constant):
Ra = Constant(0)  # Rayleigh number
approximation = BoussinesqApproximation(Ra)


steady_state_tolerance = 1e-9

# Nullspaces and near-nullspaces:
Z_nullspace = create_stokes_nullspace(Z, closed=True, rotational=False)

# Write output files in VTK format:
u, p = z.split() #subfunctions  # Do this first to extract individual velocity and pressure fields.
# Next rename for output:
u.rename("Incremental Displacement")
p.rename("Pressure")
# Create output file and select output_frequency:
output_file = File("output_viscoelastic_be.pvd")
dump_period = dt

rho0 = 4500 # density in kg/m^3
g = 10 # gravitational acceleration in m/s^2

lam = D # wavelength of load in m
k = 2 * pi / lam # wavenumber in m^-1
F0 = Constant(1000) # initial free surface amplitude in m
X = SpatialCoordinate(mesh)
eta = F0 * cos(k * X[0])
n = FacetNormal(mesh)

stokes_bcs = {
    bottom_id: {'un': 0},
    top_id: {'stress': -rho0 * g * (eta+dot(displacement,n)) * n},
    left_id: {'un': 0},
    right_id: {'un': 0},
}

up_fields = {
        }

stokes_solver = StokesSolver(z, T, approximation, bcs=stokes_bcs, mu=effective_viscosity, previous_stress=previous_stress, cartesian=True)
#                             nullspace=Z_nullspace, transpose_nullspace=Z_nullspace)



output_file.write(u, displacement, p)


# Now perform the time loop:
for timestep in range(0, max_timesteps):



    # Solve Stokes sytem:
    stokes_solver.solve()
    
    u_old.assign(u)
    # Write output:
    displacement.interpolate(displacement+u)
    output_file.write(u, displacement, p)

    #time += dt
    if timestep == 100:
        F0.assign(0)





with CheckpointFile("Final_State.h5", "w") as final_checkpoint:
    final_checkpoint.save_mesh(mesh)
    final_checkpoint.save_function(z, name="Stokes")
