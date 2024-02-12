from gadopt import *
import numpy as np


def implicit_viscous_two_freesurface_model(nx, dt_factor, do_write=True, iterative_2d=False):
    # Test case from Section 3.1.2 of `An implicit free surface algorithm
    # for geodynamical simulations', Kramer et al 2012.

    # Set up geometry:
    D = 3e6  # Depth of domain in m
    lam_dimensional = D/2  # wavelength of load in m
    L = lam_dimensional  # Length of the domain in m
    L0 = D  # characteristic length scale for scaling the equations
    lam = lam_dimensional/L0  # dimensionless lambda

    ny = nx
    mesh = RectangleMesh(nx, ny, L/L0, D/L0)  # Rectangle mesh generated via firedrake
    left_id, right_id, bottom_id, top_id = 1, 2, 3, 4  # Boundary IDs

    # Set up function spaces - currently using the bilinear Q2Q1 element pair:
    V = VectorFunctionSpace(mesh, "CG", 2)  # Velocity function space (vector)
    W = FunctionSpace(mesh, "CG", 1)  # Pressure function space (scalar)
    Q = FunctionSpace(mesh, "CG", 2)  # Temperature function space (scalar)
    Z = MixedFunctionSpace([V, W, W, W])  # Mixed function space.

    # Function to store the solutions:
    z = Function(Z)  # a field over the mixed function space Z.
    u, p, eta, zeta = split(z)  # Returns symbolic UFL expression for u, p and eta
    u_, p_, eta_, zeta_ = z.subfunctions  # Returns functions for u, p and eta

    # Next rename for output:
    u_.rename("Velocity")
    p_.rename("Pressure")
    eta_.rename("eta")
    zeta_.rename("zeta")

    T = Function(Q, name="Temperature").assign(0)  # Setup a dummy function for temperature
    # Output function space information:
    log("Number of Velocity DOF:", V.dim())
    log("Number of Pressure DOF:", W.dim())
    log("Number of Velocity and Pressure DOF:", V.dim()+W.dim())
    log("Number of Temperature DOF:", Q.dim())

    # Stokes related constants (note that since these are included in UFL, they are wrapped inside Constant):
    Ra = Constant(0)  # Rayleigh number, here we set this to zero as there are no bouyancy terms
    approximation = BoussinesqApproximation(Ra)

    rho0 = approximation.rho  # This defaults to rho0 = 1 (dimensionless)
    rho_bottom = 2  # Bottom external density is 2x larger from Section 3.1.2 Kramer et al 2012
    g = approximation.g  # This defaults to g = 1 (dimensionless)

    kk = Constant(2 * pi / lam)  # wavenumber (dimensionless)
    F0 = Constant(1000 / L0)  # initial free surface amplitude (dimensionless)
    G0 = Constant(1000 / L0)  # initial free surface amplitude (dimensionless)
    X = SpatialCoordinate(mesh)
    eta_.interpolate(F0 * cos(kk * X[0]))  # Initial free surface condition
    zeta_.interpolate(G0 * cos(kk * X[0]))  # Initial free surface condition

    # timestepping
    mu = Constant(1)  # Shear modulus (dimensionless)
    tau0 = Constant(2 * kk * mu / (rho0 * g))  # Characteristic time scale (dimensionless)
    delta_rho = rho_bottom - rho0
    tau0_zeta = Constant(2 * kk * mu / (delta_rho * g))  # Characteristic time scale (dimensionless)
    log("tau0", tau0)

    dt = Constant(dt_factor*tau0)  # timestep (dimensionless)
    log("dt (dimensionless)", dt)

    time = Constant(0.0)
    max_timesteps = round(10*tau0/dt)  # Simulation runs for 10 characteristic time scales so end state is close to being fully relaxed
    log("max_timesteps", max_timesteps)

    # No normal flow except on the free surfaces
    # Free surface boundary conditions are applied automatically in stokes_integrators and momentum_equation for implicit free surface coupling
    stokes_bcs = {
        top_id: {'free_surface': {}},
        bottom_id: {'free_surface': {'exterior_density': rho_bottom}},  # Specify exterior density below bottom free surface
        left_id: {'ux': 0},
        right_id: {'ux': 0},
    }

    stokes_solver = StokesSolver(z, T, approximation, bcs=stokes_bcs, mu=mu, cartesian=True, free_surface_dt=dt, iterative_2d=iterative_2d)

    class SpecifiedNodeBC(DirichletBC):
        def __init__(self, V, value, node):
            sub_domain = 0  # I think sub_domain is not actually used in the __init__
            super().__init__(V, value, sub_domain)
            self.nodes = np.array([node])

    def find_nearest(point, V):
        m = V.mesh()
        # Now make the VectorFunctionSpace corresponding to V.
        W = VectorFunctionSpace(m, V.ufl_element())

        # Next, interpolate the coordinates onto the nodes of W.
        X = interpolate(m.coordinates, W)
        distance_vector = X.dat.data - point
        abs_distance = np.square(distance_vector[:, 0])+np.square(distance_vector[:, 1])
        idx = abs_distance.argmin()
        return idx, X.dat.data[idx]
    
#    for i in range(80):
#        stationary_point = [0.125, i*(1/80)]
#        node, coords = find_nearest(stationary_point, Z.sub(0).sub(1))
#        print(f"Nearest point is {coords}, at id {node}")
#   pin_bc = SpecifiedNodeBC(Z.sub(0), as_vector((0,0)), node)
#        pin_bc = SpecifiedNodeBC(Z.sub(0).sub(1), 0, node)
#        stokes_solver.strong_bcs.append(pin_bc)
    
    alpha = 0.1 # * (dt_factor / 2)
    stokes_solver.F += alpha * stokes_solver.test[0][1] * (stokes_solver.stokes_vars[0][1] - 0)*dx
#   stationary_point = [0.375, 0.5 * D / L0]
#    node, coords = find_nearest(stationary_point, V.sub(1))
#    print(f"Nearest point is {coords}, at id {node}")
#    pin_bc2 = SpecifiedNodeBC(Z.sub(0), as_vector((0,0)), node)
#    pin_bc2 = SpecifiedNodeBC(Z.sub(0).sub(1), 0, node)
#    print("pin bc value is ", pin_bc.function_arg)
#    stokes_solver.strong_bcs.append(pin_bc2)

    if do_write:
        eta_midpoint = []
        eta_midpoint.append(eta_.at((L/L0)/2, (D/L0)-0.001/L0))

    # analytical function
    eta_analytical = Function(W, name="eta analytical")
    eta_analytical.interpolate(exp(-time/tau0)*F0 * cos(kk * X[0]))

    zeta_analytical = Function(W, name="eta analytical")
    zeta_analytical.interpolate(exp(-time/tau0_zeta)*G0 * cos(kk * X[0]))
    # Create output file and select output_frequency:
    if do_write:
        # Write output files in VTK format:
        dump_period = 1
        log("dump_period ", dump_period)
        filename = "implicit_viscous_freesurface_topbot_innerfs1_1e-10"
        output_file = File(f"{filename}_D{float(D/L0)}_mu{float(mu)}_nx{nx}_dt{float(dt/tau0)}tau.pvd")
        output_file.write(u_, eta_, p_, eta_analytical, zeta_, zeta_analytical)

    error = 0
    error_zeta = 0
    # Now perform the time loop:
    for timestep in range(1, max_timesteps+1):

        # Solve Stokes sytem:
        stokes_solver.solve()
        
        time.assign(time + dt)
        eta_analytical.interpolate(exp(-time/tau0)*F0 * cos(kk * X[0]))
        zeta_analytical.interpolate(exp(-time/tau0_zeta)*G0 * cos(kk * X[0]))

        local_error = assemble(pow(eta-eta_analytical, 2)*ds(top_id))
        error += local_error*dt
        local_error_zeta = assemble(pow(zeta-zeta_analytical, 2)*ds(bottom_id))
        error_zeta += local_error_zeta*dt

        # Write output:
        if do_write:
            eta_midpoint.append(eta_.at((L/L0)/2, (D/L0)-0.001/L0))

            if timestep % dump_period == 0:
                log("timestep", timestep)
                log("time", float(time))
                output_file.write(u_, eta_, p_, eta_analytical, zeta_, zeta_analytical)
    if do_write:
        with open(f"{filename}_D{float(D/L0)}_mu{float(mu)}_nx{nx}_dt{float(dt/tau0)}tau.txt", 'w') as file:
            for line in eta_midpoint:
                file.write(f"{line}\n")

    final_error = pow(error, 0.5)/L
    final_error_zeta = pow(error_zeta, 0.5)/L
    return final_error, final_error_zeta


if __name__ == "__main__":
    # default case run with nx = 80 for four dt factors
    dt_factors = 2 / (2**np.arange(5))
    # errors = np.array([implicit_viscous_two_freesurface_model(80, dtf) for dtf in dt_factors])
    # np.savetxt("errors-implicit-top-free-surface-coupling.dat", errors[:, 0])
    # np.savetxt("errors-implicit-bottom-free-surface-coupling.dat", errors[:, 1])

    errors_iterative = np.array([implicit_viscous_two_freesurface_model(80, dtf, iterative_2d=True) for dtf in dt_factors])
    np.savetxt("errors-implicit-iterative-top-free-surface-coupling.dat", errors_iterative[:, 0])
    np.savetxt("errors-implicit-iterative-bottom-free-surface-coupling.dat", errors_iterative[:, 1])
