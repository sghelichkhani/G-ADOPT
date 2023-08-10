from gadopt import *
from mpi4py import MPI
import numpy as np
import pandas as pd

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


eta = Function(W)

T = Function(Q, name="Temperature").assign(0)
# Output function space information:
log("Number of Velocity DOF:", V.dim())
log("Number of Pressure DOF:", W.dim())
log("Number of Velocity and Pressure DOF:", V.dim()+W.dim())
log("Number of Temperature DOF:", Q.dim())





eta_eq = FreeSurfaceEquation(W, W, surface_id=top_id)

steady_state_tolerance = 1e-9

# Nullspaces and near-nullspaces:
Z_nullspace = create_stokes_nullspace(Z, closed=True, rotational=False)
#eta_nullspace=VectorSpaceBasis(constant=True)

# Write output files in VTK format:
u_, p_ = z.subfunctions #subfunctions  # Do this first to extract individual velocity and pressure fields.
# Next rename for output:
u_.rename("Velocity")
p_.rename("Pressure")
# Create output file and select output_frequency:
output_file = File("/data/free_surface/2d_box/viscous/output_viscous_dt_0.5tau.pvd")

# Stokes related constants (note that since these are included in UFL, they are wrapped inside Constant):
Ra = Constant(0)  # Rayleigh number
rho0 = 4500 # density in kg/m^3
g = 10 # gravitational acceleration in m/s^2
approximation = BoussinesqApproximation(Ra,g=g,rho=rho0)

lam = D/2 # wavelength of load in m
kk = 2 * pi / lam # wavenumber in m^-1
F0 = 1000 # initial free surface amplitude in m
X = SpatialCoordinate(mesh)
eta.interpolate(F0 * cos(kk * X[0]))
n = FacetNormal(mesh)
# timestepping 

mu = 1e21 # Shear modulus in Pa
tau0 = 2 * kk * mu / (rho0 * g) 
print("tau0", tau0)
dt = 0.5*tau0  # Initial time-step
dump_period = round(tau0/dt)
print(dump_period)
time = 0.0
max_timesteps = round(10*tau0/dt)
print("max_timesteps", max_timesteps)
stokes_bcs = {
    bottom_id: {'un': 0},
    top_id: {'stress': -rho0 * g * eta * n},
    left_id: {'un': 0},
    right_id: {'un': 0},
}

eta_fields = {'velocity': u_,
               'surface_id': top_id}

#class InteriorBC(DirichletBC):
#    """DirichletBC applied to anywhere that is *not* on the specified boundary"""
#    @utils.cached_property
#    def nodes(self):
#        return np.array(list(set(range(self._function_space.node_count)) - set(super().nodes)))

eta_bcs = {}

eta_strong_bcs = [InteriorBC(W, 0., top_id)]

mumps_solver_parameters = {
    'snes_monitor': None,
    'snes_type': 'ksponly',
    'ksp_type': 'preonly',
    'pc_type': 'lu',
    'pc_factor_mat_solver_type': 'mumps',
    'mat_type': 'aij',
    'snes_max_it': 100,
    'snes_rtol': 1e-8,
    'snes_atol': 1e-6,
    'mat_mumps_icntl_14': 200 
 #   'ksp_monitor': None,
}
stokes_solver = StokesSolver(z, T, approximation, bcs=stokes_bcs, mu=mu, cartesian=True)
#                             nullspace=Z_nullspace, transpose_nullspace=Z_nullspace)


eta_timestepper = BackwardEuler(eta_eq, eta, eta_fields, dt, bnd_conditions=eta_bcs, strong_bcs=eta_strong_bcs,solver_parameters=mumps_solver_parameters) #,nullspace=eta_nullspace)

output_file.write(u_, eta, p_)

# evaluate surface
surface_displacement_df = pd.DataFrame()
outfile="/data/free_surface/2d_box/viscous/eta_dt0.03125tau.csv"
intervals = nx*4+1
x = np.linspace(0, D, intervals)
points = []

for xi in x:
    points.append([xi, D])
print("points", points)

vom = VertexOnlyMesh(mesh, points)
P0DG = FunctionSpace(vom, "DG", 0)

surface_displacement = Function(P0DG).interpolate(eta)
#surface_displacement_df["eta_t_ " + str(time / tau0)] = surface_displacement.dat.data
#surface_displacement_df.to_csv(outfile)

with open(outfile, "a") as f:
#        f.write("\n")
        np.savetxt(f, surface_displacement.dat.data,delimiter=",")

# analytical function

eta_analytical = Function(W)

eta_analytical.interpolate(exp(-time/tau0)*F0 * cos(kk * X[0]))

error = 0
# Now perform the time loop:
for timestep in range(1, max_timesteps+1):


    #print("u_ before: ", u_.dat.data[:])
    #print("eta u.. beore: ", eta_fields['velocity'].dat.data[:])
    # Solve Stokes sytem:
    stokes_solver.solve()
    eta_timestepper.advance(time)
    
    eta_analytical.interpolate(exp(-time/tau0)*F0 * cos(kk * X[0]))
    
    local_error = assemble(pow(eta-eta_analytical,2)*ds(top_id))
    error += local_error*dt
    #print("u_ after: ", u_.dat.data[:])
    #print("eta u.. after: ", eta_fields['velocity'].dat.data[:])
    # Write output:

    time += dt
    if timestep % dump_period == 0:
        print("timestep", timestep)
        print("time", time)
        output_file.write(u_, eta, p_)
        surface_displacement = Function(P0DG).interpolate(eta)
        print(surface_displacement.dat.data)
        with open(outfile, "ab") as f:
            f.write(b"\n")
            np.savetxt(f, surface_displacement.dat.data,delimiter=",")



print("final error", pow(error,0.5))



with CheckpointFile("Final_State.h5", "w") as final_checkpoint:
    final_checkpoint.save_mesh(mesh)
    final_checkpoint.save_function(z, name="Stokes")
