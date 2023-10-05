# box model based on weerdestejin et al 2023

from gadopt import *
from mpi4py import MPI
import os
import numpy as np
OUTPUT=True
output_directory="/data/free_surface/2d_box/viscoelastic/aspect_box/"

    
# Set up geometry:
dx = 150e3  # horizontal grid resolution
dz = 150e3  # vertical grid resolution
L = 1500e3  # length of the domain in m


# layer properties from spada et al 2011
radius_values = [6371e3, 6301e3, 5951e3, 5701e3, 3480e3]


thickness_values = [70e3, 350e3, 250e3, 2221e3, 3480e3]

density_values = [3037, 3438, 3871, 4978, 10750]

shear_modulus_values = [0.50605e11, 0.70363e11, 1.05490e11,2.28340e11, 0]

viscosity_values = [1e40, 1e21, 1e21, 2e21, 0]

D = radius_values[0]-radius_values[-1]

nx = round(L/dx)
nz = round(D/dz)

mesh = BoxMesh(nx, nx, nz, L, L, D) #, hexahedral=True)  # Rectangle mesh generated via firedrake
mesh.coordinates.dat.data[:, 2] -= D
x,y, z = SpatialCoordinate(mesh)

bottom_id, top_id = 5, 6  # Boundary IDs

# Set up function spaces - currently using the bilinear Q2Q1 element pair:
V = VectorFunctionSpace(mesh, "CG", 2)  # Displacement function space (vector)
W = FunctionSpace(mesh, "CG", 1)  # Pressure function space (scalar)
Q = FunctionSpace(mesh, "CG", 2)  # Temperature function space (scalar)
Q3 = FunctionSpace(mesh, "CG", 3)  # Temperature function space (scalar)
M = MixedFunctionSpace([V, W])  # Mixed function space.

# Function to store the solutions:
m = Function(M)  # a field over the mixed function space M.
u, p = split(m)  # Returns symbolic UFL expression for u and p

u_old = Function(V, name="u old")
u_old.assign(0)
displacement = Function(V, name="displacement").assign(0)

#eta_surf = Function(W, name="eta")
#eta_eq = FreeSurfaceEquation(W, W, surface_id=top_id)

T = Function(Q, name="Temperature").assign(0)
# Output function space information:
log("Number of Velocity DOF:", V.dim())
log("Number of Pressure DOF:", W.dim())
log("Number of Velocity and Pressure DOF:", V.dim()+W.dim())
log("Number of Temperature DOF:", Q.dim())

# timestepping

rho_ice = 931
g= 9.8125  # there is also a list but Aspect doesnt use...

viscosity = Function(W, name="viscosity")
for i in range(0,len(viscosity_values)-1):
    viscosity.interpolate(
            conditional(z >= radius_values[i+1] - radius_values[0],
                conditional( z <= radius_values[i] - radius_values[0],
                    viscosity_values[i], viscosity), viscosity))
shear_modulus = Function(W, name="shear modulus")
for i in range(0,len(shear_modulus_values)-1):
    shear_modulus.interpolate(
            conditional(z >= radius_values[i+1] - radius_values[0],
                conditional( z <= radius_values[i] - radius_values[0],
                    shear_modulus_values[i], shear_modulus), shear_modulus))

maxwell_time = viscosity / shear_modulus

n = FacetNormal(mesh)

time = Constant(0.0)

year_in_seconds = Constant(3600 * 24 * 365.25)
dt = Constant(2.5 * year_in_seconds)  # Initial time-step

dt_elastic = Constant(dt)
#    dt_elastic = conditional(dt_elastic<2*maxwell_time, 2*0.0125*tau0, dt_elastic)
#    max_timesteps = round(20*tau0/dt_elastic)

Tend = Constant(200* year_in_seconds)
max_timesteps = round(Tend/dt)
print("max timesteps", max_timesteps)
dump_period = round(Tend / (10*year_in_seconds)) #0.5*tau0/dt)
print("dump_period", dump_period)
print("dt", dt.values()[0])
#effective_viscosity = Constant(viscosity/(maxwell_time +dt_elastic/2))
#prefactor_prestress = Constant((maxwell_time - dt_elastic/2)/(maxwell_time + dt_elastic/2))
effective_viscosity = viscosity/(maxwell_time +dt)
prefactor_prestress = maxwell_time/(maxwell_time + dt)


ice_load = Function(W)

ramp = Constant(0)
Hice = 100

disc_radius = 100e3
disc = conditional(pow(x, 2)+ pow(y, 2) < pow(disc_radius,2), 1, 0)

ice_load.interpolate(ramp * rho_ice * g *Hice* disc)




u_, p_ = m.subfunctions

TP1 = TensorFunctionSpace(mesh, "CG", 1)
previous_stress = Function(TP1, name='previous_stress')
deviatoric_stress = Function(TP1, name='deviatoric_stress')
averaged_deviatoric_stress = Function(TP1, name='averaged deviatoric_stress')
# previous_stress = prefactor_prestress *  2 * effective_viscosity * sym(grad(u_old))

# Stokes related constants (note that since these are included in UFL, they are wrapped inside Constant):
Ra = Constant(0)  # Rayleigh number
approximation = BoussinesqApproximation(Ra)


steady_state_tolerance = 1e-9

# Nullspaces and near-nullspaces:

# Write output files in VTK format:
# Next rename for output:
u_.rename("Incremental Displacement")
p_.rename("Pressure")
# Create output file and select output_frequency:
filename=os.path.join(output_directory, "viscoelastic")
if OUTPUT:
    output_file = File(filename+"_weerdesteijn_aspectbox_nx"+str(nx)+"_dt"+str(round(dt/year_in_seconds))+"years_dtout_10years_Tend200years_withprestressadv_fixramp/out.pvd")
stokes_bcs = {
    bottom_id: {'un': 0},
#        top_id: {'stress': -rho0 * g * (eta + dot(displacement, n)) * n},
#        top_id: {'stress': -rho0 * g * eta * n},
    top_id: {'stress': -ice_load*n },
    #    top_id: {'old_stress': prefactor_prestress*(-rho0 * g * (eta + dot(displacement,n)) * n)},
    1: {'un': 0},
    2: {'un': 0},
    3: {'un': 0},
    4: {'un': 0},
}

up_fields = {}
stokes_fields = {
    'surface_id': 4,  # VERY HACKY!
    'previous_stress': previous_stress,  # VERY HACKY!
    'displacement': displacement,
    'rhog': density_values[0]*g}  # Incredibly hacky! rho*g

eta_fields = {'velocity': u_/dt,
                'surface_id': top_id}

eta_bcs = {} 

eta_strong_bcs = [InteriorBC(W, 0., top_id)]
stokes_solver = StokesSolver(m, T, approximation, bcs=stokes_bcs, mu=effective_viscosity, equations=ViscoElasticEquations,
                             cartesian=True, additional_fields=stokes_fields)

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

#eta_timestepper = BackwardEuler(eta_eq, eta_surf, eta_fields, dt, bnd_conditions=eta_bcs, strong_bcs=eta_strong_bcs,solver_parameters=mumps_solver_parameters) 
#    stokes_solver.fields['source'] = div(previous_stress)
# analytical function
#eta_analytical = Function(Q3, name="eta analytical").interpolate(F0-eta)
if OUTPUT:
    output_file.write(u_, u_old, displacement, p_, previous_stress, shear_modulus, viscosity)

eta_midpoint =[]
#eta_midpoint.append(displacement.at(L/2+100, -0.001)[1])

error = 0
# Now perform the time loop:
for timestep in range(1, max_timesteps+1):#int(max_timesteps/2)+1):

    ramp.assign(conditional(time < 100*year_in_seconds, 
        time/ (100*year_in_seconds), 1))
    print(ramp.values()[0]) 
    ice_load.interpolate(ramp * rho_ice * g *Hice* disc)

    stokes_solver.solve()
 #   eta_timestepper.advance(time)

    u_old.assign(u_)  # (1-dt/dt_elastic)*u_old + (dt/dt_elastic)*u)
    #u_old.assign((1-dt/dt_elastic)*u_old + (dt/dt_elastic)*u_)
    deviatoric_stress.interpolate(2 * effective_viscosity * sym(grad(u_old))+prefactor_prestress*deviatoric_stress)
#        averaged_deviatoric_stress.interpolate((1-dt/dt_elastic)*averaged_deviatoric_stress + (dt/dt_elastic)*deviatoric_stress)
    previous_stress.interpolate(prefactor_prestress*deviatoric_stress)  # most recent without elastic prestress
#        previous_stress.interpolate(prefactor_prestress*averaged_deviatoric_stress)  # try elastic timestep
    #previous_stress.interpolate((dt/dt_elastic)*(prefactor_prestress* 2 * effective_viscosity * sym(grad(u_old))+prefactor_prestress*previous_stress)+(1-dt/dt_elastic)*previous_stress)

    displacement.interpolate(displacement+u)
 #   eta_midpoint.append(displacement.at(L/2, -0.001)[1])
#    eta_midpoint.append(eta_surf.at(L/2, -0.001))

#        Vc = mesh.coordinates.function_space()
#        x, y = SpatialCoordinate(mesh)
#        f = Function(Vc).interpolate(as_vector([x+u_[0], y+u_[1]]))
#        mesh.coordinates.assign(f)


#        if timestep ==2:
#            with open(filename+"_D3e6_visc1e21_shearmod1e11_nx"+str(nx)+"_dt"+str(dt_factor)+"tau_be_a4_ny"+str(ny)+"_lam"+str(lam)+"_L"+str(L)+"_prestressadvsurf.txt", 'w') as file:
#                for line in eta_midpoint:
#                    file.write(f"{line}\n")
#            return 1
    
    time.assign(time+dt)
    
#    if timestep >= round(max_timesteps/2):
#        eta_analytical.interpolate(exp(-(time-10*tau0)/tau0)*h0 * (cos(kk * X[0]))*(1 - rho0*g/(2*kk*shear_modulus)))
#        #eta_analytical.interpolate((-exp(-(time-10*tau0)/tau0))*h0 * cos(kk * X[0])) #rho0*g/(2*kk*shear_modulus)))
#            local_error = assemble(pow(displacement[1]-eta_analytical,2)*ds(top_id))
#        local_error = assemble(pow(eta_surf-eta_analytical,2)*ds(top_id))
#        error += local_error*dt.values()[0]

    
#            output_file.write(u_, u_old, displacement, p_, previous_stress, eta_analytical)
    # Write output:
    if timestep % dump_period == 0:
        print("timestep", timestep)
        print("time", time.values()[0])
        if OUTPUT:
            output_file.write(u_, u_old, displacement, p_, previous_stress, shear_modulus, viscosity)
    
#with open(filename+"_D3e6_visc1e21_shearmod1e11_nx"+str(nx)+"_dt"+str(dt_factor)+"tau_a6_refinemesh_nosurfadv_expfreesurface.txt", 'w') as file:
#    for line in eta_midpoint:
#        file.write(f"{line}\n")
#final_error = pow(error,0.5)/L





