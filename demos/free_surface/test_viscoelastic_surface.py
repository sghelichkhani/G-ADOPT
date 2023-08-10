from gadopt import *
from mpi4py import MPI
import os
import numpy as np
OUTPUT=True
output_directory="."

def viscoelastic_model(nx, dt_factor):
    
    # Set up geometry:
    ny = nx
    L = 3e6  # length of domain in m
    D = L  # Depth of the domain in m
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

    u_old = Function(V, name="u old")
    u_old.assign(0)
    displacement = Function(V, name="displacement").assign(0)

    T = Function(Q, name="Temperature").assign(0)
    # Output function space information:
    log("Number of Velocity DOF:", V.dim())
    log("Number of Pressure DOF:", W.dim())
    log("Number of Velocity and Pressure DOF:", V.dim()+W.dim())
    log("Number of Temperature DOF:", Q.dim())

    # timestepping
    rho0 = 4500  # density in kg/m^3
    g = 10  # gravitational acceleration in m/s^2
    viscosity = Constant(1e21)  # Viscosity Pa s
    shear_modulus = Constant(1e11)  # Shear modulus in Pa
    maxwell_time = viscosity / shear_modulus

    lam = D/4  # wavelength of load in m
    kk = 2 * pi / lam  # wavenumber in m^-1
    F0 = Constant(1000)  # initial free surface amplitude in m
    X = SpatialCoordinate(mesh)
    eta = F0 * (1-cos(kk * X[0]))
    n = FacetNormal(mesh)
    tau0 = Constant(2 * kk * viscosity / (rho0 * g))
    print("tau0", tau0.values()[0])
    time = Constant(0.0)
    dt = Constant(dt_factor * tau0)  # Initial time-step
    # max_timesteps = round(50*maxwell_time/dt)
    max_timesteps = round(20*tau0/dt)
    print("max timesteps", max_timesteps)
    dump_period = round(0.5*tau0/dt)
    print("dump_period", dump_period)

    # dt = Constant(maxwell_time/4)  # Initial time-step
    dt_elastic = Constant(dt*3)
    print("dt", dt.values()[0])
    effective_viscosity = Constant(viscosity/(maxwell_time+0.5*dt))
    prefactor_prestress = Constant((maxwell_time -0.5*dt)/(maxwell_time + 0.5*dt))

    print("effective visc", effective_viscosity.values()[0])
    print("prefactor prestress", prefactor_prestress.values()[0])

    u_, p_ = z.split()

    TP1 = TensorFunctionSpace(mesh, "CG", 1)
    previous_stress = Function(TP1, name='previous_stress').interpolate(prefactor_prestress * 2 * effective_viscosity * sym(grad(u_old)))
    # previous_stress = prefactor_prestress *  2 * effective_viscosity * sym(grad(u_old))

    # Stokes related constants (note that since these are included in UFL, they are wrapped inside Constant):
    Ra = Constant(0)  # Rayleigh number
    approximation = BoussinesqApproximation(Ra)


    steady_state_tolerance = 1e-9

    # Nullspaces and near-nullspaces:
    Z_nullspace = create_stokes_nullspace(Z, closed=True, rotational=False)

    # Write output files in VTK format:
    # Next rename for output:
    u_.rename("Incremental Displacement")
    p_.rename("Pressure")
    # Create output file and select output_frequency:
    filename=os.path.join(output_directory, "viscoelastic")
    output_file = File(filename+"_D3e6_visc1e21_shearmod1e11_nx"+str(nx)+"_dt"+str(dt_factor)+"tau.pvd")

    stokes_bcs = {
        bottom_id: {'un': 0},
        top_id: {'stress': rho0 * g * (-eta - dot(displacement, n)) * n},
        #    top_id: {'old_stress': prefactor_prestress*(-rho0 * g * (eta + dot(displacement,n)) * n)},
        left_id: {'un': 0},
        right_id: {'un': 0},
    }

    up_fields = {}
    stokes_fields = {
        'surface_id': 4,  # VERY HACKY!
        'previous_stress': previous_stress,  # VERY HACKY!
        'rhog': -45000}  # Incredibly hacky! rho*g

    stokes_solver = StokesSolver(z, T, approximation, bcs=stokes_bcs, mu=effective_viscosity, equations=ViscoElasticEquations,
                                 cartesian=True, additional_fields=stokes_fields)

#    stokes_solver.fields['source'] = div(previous_stress)
    # analytical function
    eta_analytical = Function(Q, name="eta analytical").interpolate(F0-eta)
    h0 = Constant(F0)
    if OUTPUT:
        output_file.write(u_, u_old, displacement, p_, previous_stress, eta_analytical)

    error = 0
    # Now perform the time loop:
    for timestep in range(1, max_timesteps+1):#int(max_timesteps/2)+1):
        if timestep == max_timesteps/2:
            F0.assign(0)

        
        stokes_solver.solve()

        u_old.assign(u_)  # (1-dt/dt_elastic)*u_old + (dt/dt_elastic)*u)
        previous_stress.interpolate(prefactor_prestress * 2 * effective_viscosity * sym(grad(u_old))+prefactor_prestress*previous_stress)

        displacement.interpolate(displacement+u)
        
        time.assign(time+dt)
        if timestep >= round(max_timesteps/2):
            eta_analytical.interpolate(exp(-(time-10*tau0)/tau0)*h0 * (cos(kk * X[0]))*(1 - 0.0568)) #rho0*g/(2*kk*shear_modulus)))
            #eta_analytical.interpolate((-exp(-(time-10*tau0)/tau0))*h0 * cos(kk * X[0])) #rho0*g/(2*kk*shear_modulus)))
            local_error = assemble(pow(displacement[1]-eta_analytical,2)*ds(top_id))
            error += local_error*dt.values()[0]

        
#            output_file.write(u_, u_old, displacement, p_, previous_stress, eta_analytical)
        # Write output:
        if timestep % dump_period == 0:
            print("timestep", timestep)
            print("time", time.values()[0])
            if OUTPUT:
                output_file.write(u_, u_old, displacement, p_, previous_stress, eta_analytical)
        
    
    final_error = pow(error,0.5)/L
    return final_error


dt_factors = [0.1, 0.05, 0.025, 0.0125, 0.00625, 0.003125]#, 0.125, 0.0625, 0.03125]
errors = np.array([viscoelastic_model(40, dtf) for dtf in dt_factors]) 
conv = np.log(errors[:-1]/errors[1:])/np.log(2)

print('time surface displacement errors: ', errors[:])
print('time surface displacement conv: ', conv[:])


