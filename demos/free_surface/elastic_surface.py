from gadopt import *
from mpi4py import MPI

# Set up geometry:
nx, ny = 160, 160
L = 3e6 # length of domain in m
D = L # Depth of the domain in m
mesh = RectangleMesh(nx, ny, L, D)  # Rectangle mesh generated via firedrake
left_id, right_id, bottom_id, top_id = 1, 2, 3, 4  # Boundary IDs

# Set up function spaces - currently using the bilinear Q2Q1 element pair:
V = VectorFunctionSpace(mesh, "CG", 2)  # Velocity function space (vector)
W = FunctionSpace(mesh, "CG", 1)  # Pressure function space (scalar)
Q = FunctionSpace(mesh, "CG", 2)  # Temperature function space (scalar)
Z = MixedFunctionSpace([V, W])  # Mixed function space.

# Function to store the solutions:
z = Function(Z)  # a field over the mixed function space Z.
u, p = split(z)  # Returns symbolic UFL expression for u and p

T = Function(Q, name="Temperature").assign(0)
# Output function space information:
log("Number of Velocity DOF:", V.dim())
log("Number of Pressure DOF:", W.dim())
log("Number of Velocity and Pressure DOF:", V.dim()+W.dim())
log("Number of Temperature DOF:", Q.dim())


# Stokes related constants (note that since these are included in UFL, they are wrapped inside Constant):
Ra = Constant(0)  # Rayleigh number
approximation = BoussinesqApproximation(Ra)

mu = 1e11 # Shear modulus in Pa

steady_state_tolerance = 1e-9

# Nullspaces and near-nullspaces:
Z_nullspace = create_stokes_nullspace(Z, closed=True, rotational=False)

# Write output files in VTK format:
u, p = z.split() #subfunctions  # Do this first to extract individual velocity and pressure fields.
# Next rename for output:
u.rename("Displacement")
p.rename("Pressure")
# Create output file and select output_frequency:
output_file = File("output_elastic_0.25D_mu1e11_nonull_noprestressadv_posiceheight.pvd")

rho0 = 4500 # density in kg/m^3
g = 10 # gravitational acceleration in m/s^2

lam = D/4 # wavelength of load in m
k = 2 * pi / lam # wavenumber in m^-1
F0 = 1000 # initial free surface amplitude in m
X = SpatialCoordinate(mesh)
eta = F0 * (1 - cos(k * X[0]))
n = FacetNormal(mesh)

stokes_bcs = {
    bottom_id: {'un': 0},
    top_id: {'stress': -rho0 * g * eta * n},
    left_id: {'un': 0},
    right_id: {'un': 0},
}
stokes_fields = {
        'surface_id': 4, # surface id for prestress advection
        'rhog': 45000 # rho*g for prestress advection
         }

stokes_solver = StokesSolver(z, T, approximation, bcs=stokes_bcs, mu=mu,
                             cartesian=True, equations=ElasticPrestressEquations, additional_fields=stokes_fields)
  #                           nullspace=Z_nullspace, transpose_nullspace=Z_nullspace)



output_file.write(u, p)


# Solve Stokes sytem:
stokes_solver.solve()


output_file.write(u, p)
# Compute diagnostics:
bcu = DirichletBC(u.function_space(), 0, top_id)
ux_max = u.dat.data_ro_with_halos[bcu.nodes, 0].max(initial=0)
ux_max = u.comm.allreduce(ux_max, MPI.MAX)  # Maximum Vx at surface




with CheckpointFile("Final_State.h5", "w") as final_checkpoint:
    final_checkpoint.save_mesh(mesh)
    final_checkpoint.save_function(z, name="Stokes")
