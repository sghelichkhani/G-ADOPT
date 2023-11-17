from optimisation import OptimisationFunction, L_BFGS_BOptimizer
from gadopt import *
from gadopt.inverse import *
import warnings
warnings.filterwarnings("ignore")


def helmholtz(V, source):
    u = Function(V)
    v = TestFunction(V)
    F = inner(grad(v), grad(u)) * dx + 100.0*v*u*dx - v*source*dx

    solve(F == 0, u)
    return u


mesh = UnitIntervalMesh(10)
# create a checkpointable mesh by writing to disk and restoring
with CheckpointFile("mesh_helmholtz.h5", "w") as f:
    f.save_mesh(mesh)
with CheckpointFile("mesh_helmholtz.h5", "r") as f:
    mesh = f.load_mesh("firedrake_default")

V = FunctionSpace(mesh, "CG", 1)
source_ref = Function(V, name="source_ref")
x = SpatialCoordinate(mesh)
source_ref.interpolate(cos(pi * x**2))

with stop_annotating():
    # compute reference solution
    u_ref = helmholtz(V, source_ref)
u_ref.rename("u_ref")

source = Function(V)
c = Control(source)
# tape the forward solution
u = helmholtz(V, source)
u.rename("u")
File("input.pvd").write(u, u_ref, source)

J = assemble(1e6 * (u - u_ref)**2 * dx)
rf = ReducedFunctional(J, c)

T_lb = Function(V, name="Lower bound")
T_ub = Function(V, name="Upper bound")
T_lb.assign(-1.0)
T_ub.assign(1.0)

stop_annotating()

opt_f = OptimisationFunction(cntrl=c)
optimiser = L_BFGS_BOptimizer(rf, opt_f, bounds=[(T_lb, T_ub)], m=2)
optimiser.optimize()