from gadopt import *
from mpi4py import MPI
import os
import numpy as np
OUTPUT=True
output_directory="."

def elastic_model(nx):


    # Set up geometry:


    ny = nx
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

    filename=os.path.join(output_directory, "elastic")
    # Create output file and select output_frequency:
    output_file = File(filename+"_D"+str(D)+"_mu"+str(mu)+"_nx"+str(nx)+".pvd")

    rho0 = 4500 # density in kg/m^3
    g = 10 # gravitational acceleration in m/s^2

    lam = D/8 # wavelength of load in m # better convergence (~2.9) than D or D/4
    k = 2 * pi / lam # wavenumber in m^-1
    F0 = 1000 # initial free surface amplitude in m
    X = SpatialCoordinate(mesh)
    eta = F0 * cos(k * X[0])
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
                                 cartesian=True, additional_fields=stokes_fields)

    # analytical function
    w_surface_analytical = Function(Q)  # analytical solution for vertical displacement
    w_surface_analytical.interpolate((-rho0*g*F0/(2*mu*k)) * cos(k * X[0]))

    if OUTPUT:
        output_file.write(u, p, w_surface_analytical)


    # Solve Stokes sytem:
    stokes_solver.solve()


    error = pow(assemble(pow(u[1]-w_surface_analytical,2)*ds(top_id)),0.5)/L

    if OUTPUT:
        output_file.write(u, p, w_surface_analytical)
    
    return error



cells = [20,40,80,160]

errors = np.array([elastic_model(c) for c in cells]) 
conv = np.log(errors[:-1]/errors[1:])/np.log(2)

print('surface displacement errors: ', errors[:])
print('surface displacement conv: ', conv[:])
assert all(conv[:]> 2.8)
