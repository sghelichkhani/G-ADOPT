import numpy as np
import shapely as sl
from mpi4py import MPI

from gadopt import *


def cosine_curve(x, amplitude, wavelength, vertical_shift):
    return amplitude * np.cos(2 * np.pi / wavelength * x) + vertical_shift


nx, ny = 64, 64
lx, ly = 0.9142, 1

mesh = RectangleMesh(nx, ny, lx, ly, quadrilateral=True)
mesh.cartesian = True
left_id, right_id, bottom_id, top_id = 1, 2, 3, 4

V = VectorFunctionSpace(mesh, "Q", 2)
W = FunctionSpace(mesh, "Q", 1)
Z = MixedFunctionSpace([V, W])
K = FunctionSpace(mesh, "DQ", 2)
R = FunctionSpace(mesh, "R", 0)

z = Function(Z)
u, p = split(z)
z.subfunctions[0].rename("Velocity")
z.subfunctions[1].rename("Pressure")
psi = Function(K, name="Level set")

interface_deflection = 0.1
interface_wavelength = 2 * lx
material_interface_y = 0.2

isd_params = (interface_deflection, interface_wavelength, material_interface_y)

interface_x = np.linspace(0, lx, 1000)
interface_y = cosine_curve(interface_x, *isd_params)
interface = sl.LineString([*np.column_stack((interface_x, interface_y))])
sl.prepare(interface)

signed_distance = [
    (1 if y > cosine_curve(x, *isd_params) else -1) * interface.distance(sl.Point(x, y))
    for x, y in node_coordinates(psi)
]
epsilon = psi.comm.allreduce(mesh.cell_sizes.dat.data.min() / 4, MPI.MIN)
psi.dat.data[:] = (1 + np.tanh(np.asarray(signed_distance) / 2 / epsilon)) / 2

Ra_c = material_field(psi, [Ra_c_buoyant := 0, Ra_c_dense := 1], interface="sharp")
approximation = Approximation("BA", dimensional=False, parameters={"Ra_c": Ra_c})

Z_nullspace = create_stokes_nullspace(Z)

stokes_bcs = {
    bottom_id: {"u": 0},
    top_id: {"u": 0},
    left_id: {"ux": 0},
    right_id: {"ux": 0},
}

stokes_solver = StokesSolver(
    z,
    approximation,
    bcs=stokes_bcs,
    nullspace={"nullspace": Z_nullspace, "transpose_nullspace": Z_nullspace},
)
stokes_solver.solve()

delta_t = Function(R).assign(1.0)
t_adapt = TimestepAdaptor(delta_t, u, V, target_cfl=0.6)

level_set_solver = LevelSetSolver(psi, u, delta_t, eSSPRKs10p3, epsilon)

time_now, time_end = 0, 150

output_file = VTKFile("forward_output.pvd")
output_file.write(*z.subfunctions, psi, time=time_now)

step = 0
while True:
    t_adapt.maximum_timestep = time_end - time_now
    t_adapt.update_timestep()

    level_set_solver.solve(step)
    stokes_solver.solve()

    time_now += float(delta_t)
    step += 1

    output_file.write(*z.subfunctions, psi, time=time_now)

    if time_now >= time_end:
        break

with CheckpointFile("forward_checkpoint.h5", "w") as final_checkpoint:
    final_checkpoint.save_mesh(mesh)
    final_checkpoint.save_function(psi, name="Level set")
