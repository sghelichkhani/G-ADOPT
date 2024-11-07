# Idealised 2-D viscoelastic loading problem in an annulus
# =======================================================
#
# In this tutorial, we examine an idealised 2-D loading problem in an annulus domain.
#
# This example focusses on differences between running simulations in a 2-D annulus and 2-D Cartesian domain. These can be summarised as follows:
# 1. The geometry of the problem - i.e. the computational mesh.
# 2. The radial direction of gravity (as opposed to the vertical direction in a Cartesian domain).
# 3. Solving a problem with laterally varying viscosity.

# This example
# -------------
# Let's get started!
# The first step is to import the gadopt module, which
# provides access to Firedrake and associated functionality.

from gadopt import *
from gadopt.utility import step_func, vertical_component, CombinedSurfaceMeasure
import pyvista as pv
import matplotlib.pyplot as plt


from gadopt.inverse import *
tape = get_working_tape()
tape.clear_tape()

# In this tutorial we are going load the mesh created by the forward cylindrical demo in the previous tutorial. This makes it easier to load the synthetic data from the previous tutorial for our 'twin' experiment. N.b. You have to run that demo first so the checkpoint is available!

# Set up geometry:
checkpoint_file = "../2d_cylindrical/viscoelastic_loading-chk.h5"
with CheckpointFile(checkpoint_file, 'r') as afile:
    mesh = afile.load_mesh(name='surface_mesh_extruded')
bottom_id, top_id = "bottom", "top"
mesh.cartesian = False
D = 2891e3  # Depth of domain in m

# We next set up the function spaces, and specify functions to hold our solutions.

# +
# Set up function spaces - currently using the bilinear Q2Q1 element pair:
V = VectorFunctionSpace(mesh, "Q", 2)  # (Incremental) Displacement function space (vector)
W = FunctionSpace(mesh, "CG", 1)  # Pressure function space (scalar)
S = TensorFunctionSpace(mesh, "DG", 2)  # (Discontinuous) Stress tensor function space (tensor)
R = FunctionSpace(mesh, "R", 0)  # Real function space (for constants)

Z = MixedFunctionSpace([V, W])  # Mixed function space.

z = Function(Z)  # A field over the mixed function space Z.
u, p = split(z)  # Returns symbolic UFL expression for u and p
z.subfunctions[0].rename("Incremental Displacement")
z.subfunctions[1].rename("Pressure")

displacement = Function(V, name="displacement").assign(0)
stress_old = Function(S, name="stress_old").assign(0)
# -

# Let's set up the background profiles for the material properties with the same values as before. 


# +
X = SpatialCoordinate(mesh)

# layer properties from spada et al 2011
radius_values = [6371e3, 6301e3, 5951e3, 5701e3, 3480e3]
density_values = [3037, 3438, 3871, 4978]
shear_modulus_values = [0.50605e11, 0.70363e11, 1.05490e11, 2.28340e11]
viscosity_values = [2, -2, -2, -1.698970004]  # viscosity = 1e23 * 10**viscosity_values
# N.b. that we have modified the viscosity of the Lithosphere viscosity from
# Spada et al 2011 because we are using coarse grid resolution


def initialise_background_field(field, background_values, vertical_tanh_width=40e3):
    profile = background_values[0]
    sharpness = 1 / vertical_tanh_width
    depth = sqrt(X[0]**2 + X[1]**2)-radius_values[0]
    for i in range(1, len(background_values)):
        centre = radius_values[i] - radius_values[0]
        mag = background_values[i] - background_values[i-1]
        profile += step_func(depth, centre, mag, increasing=False, sharpness=sharpness)

    field.interpolate(profile)


density = Function(W, name="density")
initialise_background_field(density, density_values)

shear_modulus = Function(W, name="shear modulus")
initialise_background_field(shear_modulus, shear_modulus_values)

def bivariate_gaussian(x, y, mu_x, mu_y, sigma_x, sigma_y, rho, normalised_area=False):
    arg = ((x-mu_x)/sigma_x)**2 - 2*rho*((x-mu_x)/sigma_x)*((y-mu_y)/sigma_y) + ((y-mu_y)/sigma_y)**2
    numerator = exp(-1/(2*(1-rho**2))*arg)
    if normalised_area:
        denominator = 2*pi*sigma_x*sigma_y*(1-rho**2)**0.5
    else:
        denominator = 1
    return numerator / denominator


def setup_heterogenous_viscosity(viscosity):
    heterogenous_viscosity_field = Function(viscosity.function_space(), name='viscosity')
    antarctica_x, antarctica_y = -2e6, -5.5e6

    low_viscosity_antarctica = bivariate_gaussian(X[0], X[1], antarctica_x, antarctica_y, 1.5e6, 0.5e6, -0.4)
    heterogenous_viscosity_field.interpolate(-3*low_viscosity_antarctica + viscosity * (1-low_viscosity_antarctica))

    llsvp1_x, llsvp1_y = 3.5e6, 0
    llsvp1 = bivariate_gaussian(X[0], X[1], llsvp1_x, llsvp1_y, 0.75e6, 1e6, 0)
    heterogenous_viscosity_field.interpolate(-3*llsvp1 + heterogenous_viscosity_field * (1-llsvp1))

    llsvp2_x, llsvp2_y = -3.5e6, 0
    llsvp2 = bivariate_gaussian(X[0], X[1], llsvp2_x, llsvp2_y, 0.75e6, 1e6, 0)
    heterogenous_viscosity_field.interpolate(-3*llsvp2 + heterogenous_viscosity_field * (1-llsvp2))

    slab_x, slab_y = 3e6, 4.5e6
    slab = bivariate_gaussian(X[0], X[1], slab_x, slab_y, 0.7e6, 0.35e6, 0.7)
    heterogenous_viscosity_field.interpolate(-1*slab + heterogenous_viscosity_field * (1-slab))

    high_viscosity_craton_x, high_viscosity_craton_y = 0, 6.2e6
    high_viscosity_craton = bivariate_gaussian(X[0], X[1], high_viscosity_craton_x, high_viscosity_craton_y, 1.5e6, 0.5e6, 0.2)
    heterogenous_viscosity_field.interpolate(-1*high_viscosity_craton + heterogenous_viscosity_field * (1-high_viscosity_craton))

    return heterogenous_viscosity_field


normalised_viscosity = Function(W, name="Normalised viscosity")
initialise_background_field(normalised_viscosity, viscosity_values)
normalised_viscosity = setup_heterogenous_viscosity(normalised_viscosity)

viscosity = Function(normalised_viscosity, name="viscosity").interpolate(1e23*10**normalised_viscosity)

# -

# Now let's setup the ice load. For this tutorial we will have two synthetic ice sheets. Let's put one a larger one over the South Pole, with a total horizontal extent of 40 $^\circ$ and a maximum thickness of 2 km, and a smaller one offset from the North Pole with a width of 20 $^\circ$ and a maximum thickness of 1 km. To simplify things let's keep the ice load fixed in time.

# +
rho_ice = 931
g = 9.8125

Hice1 = 1000
Hice2 = 2000
year_in_seconds = Constant(3600 * 24 * 365.25)
# Disc ice load but with a smooth transition given by a tanh profile
disc_halfwidth1 = (2*pi/360) * 10  # Disk half width in radians
disc_halfwidth2 = (2*pi/360) * 20  # Disk half width in radians
surface_dx = 200*1e3
ncells = 2*pi*radius_values[0] / surface_dx
surface_resolution_radians = 2*pi / ncells
colatitude = atan2(X[0], X[1])
disc1_centre = (2*pi/360) * 25  # centre of disc1
disc2_centre = pi  # centre of disc2
disc1 = 0.5*(1-tanh((abs(colatitude-disc1_centre) - disc_halfwidth1) / (2*surface_resolution_radians)))
disc2 = 0.5*(1-tanh((abs(abs(colatitude)-disc2_centre) - disc_halfwidth2) / (2*surface_resolution_radians)))

target_normalised_ice_thickness = Function(W, name="target normalised ice thickness")
target_normalised_ice_thickness.interpolate(disc1 + (Hice2/Hice1)*disc2)

normalised_ice_thickness = Function(W, name="normalised ice thickness")
adj_ice_file = VTKFile(f"adj_ice.pvd")
#converter = RieszL2BoundaryRepresentation(W, top_id)  # convert to surface L2 representation

control = Control(normalised_ice_thickness)
ice_load = rho_ice * g * Hice1 * normalised_ice_thickness



# -


# Let's visualise the ice thickness using pyvista, by plotting a ring outside our synthetic Earth.

# +
ice_cmap = plt.get_cmap("Blues", 25)

# Make two points at the bounds of the mesh and one at the center to
# construct a circular arc.
normal = [0, 0, 1]
polar = [radius_values[0]-surface_dx/2, 0, 0]
center = [0, 0, 0]
angle = 360.0
arc = pv.CircularArcFromNormal(center, 500, normal, polar, angle)

# Stretch line by 20%
transform_matrix = np.array(
    [
        [1.2, 0, 0, 0],
        [0, 1.2, 0, 0],
        [0, 0, 1.2, 0],
        [0, 0, 0, 1],
    ])

def add_ice(p, m, scalar="normalised ice thickness"):
      
    data = m.read()[0]  # MultiBlock mesh with only 1 block

    arc_data = arc.sample(data)

    transformed_arc_data = arc_data.transform(transform_matrix)
    #m.get_array(scalar) = 2000
    p.add_mesh(transformed_arc_data, scalars=scalar, line_width=10, clim=[0, 2], cmap=ice_cmap, scalar_bar_args={
        "title": 'Normalised ice thickness',
        "position_x": 0.2,
        "position_y": 0.8,
        "vertical": False,
        "title_font_size": 22,
        "label_font_size": 18,
        "fmt": "%.1f",
        "font_family": "arial",
        "n_labels": 5,
    })

visc_file = VTKFile('viscosity.pvd').write(normalised_viscosity)
reader = pv.get_reader("viscosity.pvd")
visc_data = reader.read()[0]  # MultiBlock mesh with only 1 block
visc_cmap = plt.get_cmap("inferno_r", 25)

def add_viscosity(p):
    p.add_mesh(
        visc_data,
        component=None,
        lighting=False,
        show_edges=False,
        cmap=visc_cmap,
        clim=[-3, 2],
        scalar_bar_args={
            "title": 'Normalised viscosity',
            "position_x": 0.2,
            "position_y": 0.1,
            "vertical": False,
            "title_font_size": 22,
            "label_font_size": 18,
            "fmt": "%.0f",
            "font_family": "arial",
        }
        
    )
    


# + tags=["active-ipynb"]
# # Read the PVD file
# updated_ice_file = VTKFile('ice.pvd').write(normalised_ice_thickness, target_normalised_ice_thickness)
# reader = pv.get_reader("ice.pvd")
#
# # Create a plotter object
# plotter = pv.Plotter(shape=(1, 2), border=False, notebook=True, off_screen=False)
# plotter.subplot(0,0)
# add_ice(plotter, reader, 'target normalised ice thickness')
# add_viscosity(plotter)
# plotter.camera_position = 'xy'
# plotter.subplot(0,1)
# add_ice(plotter, reader, 'normalised ice thickness')
# add_viscosity(plotter)
#
# plotter.camera_position = 'xy'
# plotter.show()
# # Closes and finalizes movie
# plotter.close()
# -

# Let's setup the timestepping parameters with a timestep of 200 years and an output frequency of 1000 years.

# +
# Timestepping parameters
Tstart = 0
time = Function(R).assign(Tstart * year_in_seconds)

dt_years = 250
dt = Constant(dt_years * year_in_seconds)
Tend_years = 10e3
Tend = Constant(Tend_years * year_in_seconds)
dt_out_years = 1e3
dt_out = Constant(dt_out_years * year_in_seconds)

max_timesteps = round((Tend - Tstart * year_in_seconds) / dt)
log("max timesteps: ", max_timesteps)

dump_period = round(dt_out / dt)
log("dump_period:", dump_period)
log(f"dt: {float(dt / year_in_seconds)} years")
log(f"Simulation start time: {Tstart} years")
# -

# We can now define the boundary conditions to be used in this simulation.  Let's set the bottom and side boundaries to be free slip with no normal flow $\textbf{u} \cdot \textbf{n} =0$. By passing the string `ux` and `uy`, G-ADOPT knows to specify these as Strong Dirichlet boundary conditions.
#
# For the top surface we need to specify a normal stress, i.e. the weight of the ice load, as well as indicating this is a free surface.
#
# The `delta_rho_fs` option accounts for the density contrast across the free surface whether there is ice or air above a particular region of the mantle.

# Setup boundary conditions
exterior_density =  rho_ice * normalised_ice_thickness
stokes_bcs = {top_id: {'normal_stress': ice_load, 'free_surface': {'delta_rho_fs': density - exterior_density}},
              bottom_id: {'un': 0}
              }


# We also need to specify a G-ADOPT approximation, nullspaces (and near nullspaces for solver perfomance) and finally the stokes solver as before.  For this tutorial we will use a direct solver for the matrix system, so don't need to provide the near nullspace like before. 


# +
approximation = SmallDisplacementViscoelasticApproximation(density, shear_modulus, viscosity, g=g)

Z_nullspace = create_stokes_nullspace(Z, closed=False, rotational=True)

stokes_solver = ViscoelasticStokesSolver(z, stress_old, displacement, approximation,
                                         dt, bcs=stokes_bcs, constant_jacobian=True,
                                         nullspace=Z_nullspace, transpose_nullspace=Z_nullspace)
# -

# We next set up our output, in VTK format. This format can be read by programs like pyvista and Paraview.

# +
# Create a velocity function for plotting
velocity = Function(V, name="velocity")
velocity.interpolate(z.subfunctions[0]/dt)
# Create output file
output_file = VTKFile("output.pvd")
output_file.write(*z.subfunctions, displacement, velocity)

plog = ParameterLog("params.log", mesh)
plog.log_str(
    "timestep time dt u_rms u_rms_surf ux_max disp_min disp_max"
)

checkpoint_filename = "viscoelastic_loading-chk.h5"

print(bottom_id)
print(top_id)
gd = GeodynamicalDiagnostics(z, density, bottom_id, top_id)

# Initialise a (scalar!) function for logging vertical displacement
U = FunctionSpace(mesh, "CG", 2)  # (Incremental) Displacement function space (scalar)
vertical_displacement = Function(U, name="Vertical displacement")
# -

# Now let's run the simulation! At each step we call `solve` to calculate the incremental displacement and pressure fields. This will update the displacement at the surface and stress values accounting for the time dependent Maxwell consitutive equation.

for timestep in range(max_timesteps+1):

    stokes_solver.solve()

    time.assign(time+dt)

    if timestep % dump_period == 0:
        # First output step is after one solve i.e. roughly elastic displacement
        # provided dt < maxwell time.
        log("timestep", timestep)

        velocity.interpolate(z.subfunctions[0]/dt)
        output_file.write(*z.subfunctions, displacement, velocity)

        with CheckpointFile(checkpoint_filename, "w") as checkpoint:
            checkpoint.save_function(z, name="Stokes")
            checkpoint.save_function(displacement, name="Displacement")
            checkpoint.save_function(stress_old, name="Deviatoric stress")

    vertical_displacement.interpolate(vertical_component(displacement))

    # Log diagnostics:
    plog.log_str(
        f"{timestep} {float(time)} {float(dt)} "
        f"{gd.u_rms()} {gd.u_rms_top()} {gd.ux_max(top_id)} "
        f"{vertical_displacement.dat.data.min()} {vertical_displacement.dat.data.max()}"
    )

# As we can see from the plot below there is no displacement at the final time given there is no ice load!

# +
# plot no displacement!!!
# -

# Let's load the final displacement from our previous run.

# Define the component terms of the overall objective functional
with CheckpointFile(checkpoint_file, 'r') as afile:
    target_displacement = afile.load_function(mesh, name="Displacement")

# We can calculate the misfit between the displacement at the surface between the two forward models. It's a good idea to scale the objective function by the 

# +
circumference = 2 * pi * radius_values[0] 

ds = CombinedSurfaceMeasure(mesh, degree=6)

    
displacement_error = displacement - target_displacement
displacement_scale = 50
displacement_misfit = assemble(dot(displacement_error, displacement_error) / (circumference * displacement_scale**2) * ds(top_id))
# -

# Also a good idea to add smoothing and damping

damping = assemble((normalised_ice_thickness) ** 2 /circumference * ds(top_id))
smoothing = assemble(dot(grad(normalised_ice_thickness), grad(normalised_ice_thickness)) / circumference * ds(top_id))


# finally we can construct the objective summing the terms. let's also pause annotation as we are now done with the forward terms. 

# +

alpha_smoothing = 1e12
alpha_damping = 0.1
J = displacement_misfit + alpha_damping * damping + alpha_smoothing * smoothing
log("J = ", J)
log("J type = ", type(J))
# All done with the forward run, stop annotating anything else to the tape
pause_annotation()
# -

# Let's setup some call backs to help us keep back of the inversion. 

updated_ice_thickness = Function(normalised_ice_thickness, name="updated ice thickness")
updated_ice_thickness_file = File(f"update_ice_thickness.pvd")
updated_displacement = Function(displacement, name="updated displacement")
updated_out_file = File(f"updated_out.pvd")
def eval_cb(J, m):
    log("J", J)
    circumference = 2 * pi * radius_values[0] 
    # Define the component terms of the overall objective functional
    damping = alpha_damping * assemble((normalised_ice_thickness.block_variable.checkpoint) ** 2 /circumference * ds(top_id))
    smoothing = alpha_smoothing * assemble(dot(grad(normalised_ice_thickness.block_variable.checkpoint), grad(normalised_ice_thickness.block_variable.checkpoint)) / circumference * ds(top_id))
    log("damping", damping)
    log("smoothing", smoothing)

    # Write out values of control and final forward model results
    updated_ice_thickness.assign(m)
    updated_ice_thickness_file.write(updated_ice_thickness, target_normalised_ice_thickness)
    updated_displacement.interpolate(displacement.block_variable.checkpoint) 
    updated_out_file.write(updated_displacement, target_displacement)



Define reduced functional

reduced_functional = ReducedFunctional(J, control, eval_cb_post=eval_cb)

Good check that running the Forward model again using the reduced functional to check tape recorded everything accurately. 

log("J", J)
log("replay tape RF", reduced_functional(normalised_ice_thickness))

# We can calculate the derivative and plot it.

# +


dJdm = reduced_functional.derivative()

# plot derivative! 

# +
# some text about derivative...
# -

# A good way to verify this is correct is taylor test... 

# +
h = Function(normalised_ice_thickness)
h.dat.data[:] = np.random.random(h.dat.data_ro.shape)

taylor_test(reduced_functional, normalised_ice_thickness, h)

# -

# Now let's start inversion. first of all define some bounds. lower bound of zero is important. 

# +
ice_thickness_lb = Function(normalised_ice_thickness.function_space(), name="Lower bound ice thickness")
ice_thickness_ub = Function(normalised_ice_thickness.function_space(), name="Upper bound ice thickness")
ice_thickness_lb.assign(0.0)
ice_thickness_ub.assign(5)

bounds = [ice_thickness_lb, ice_thickness_ub]
# -

# setup a pyadjoint minimization problem

# +

minimisation_problem = MinimizationProblem(reduced_functional, bounds=bounds)
minimisation_parameters["Step"]["Trust Region"]["Initial Radius"] = 1e4
minimisation_parameters["Step"]["Trust Region"]["Maximum Radius"] = 1e30

optimiser = LinMoreOptimiser(
    minimisation_problem,
    minimisation_parameters,
    checkpoint_dir=f"optimisation_checkpoint",
)
# Restart file for optimisation...
updated_ice_thickness_file = File(f"update_ice_thickness.pvd")
updated_out_file = File(f"updated_out.pvd")


# +
optimiser.run()

# If we're performing mulitple successive optimisations, we want
# to ensure the annotations are switched back on for the next code
# to use them
continue_annotation()

# -

# plot misfit results. plot final displacement at different iterations.

# Looking at the animation, we can see that the weight of the ice load deforms the mantle, sinking beneath the ice load and pushing up material away from the ice load. This forebulge grows through the simulation and by 10,000 years is close to isostatic equilibrium. As the ice load is applied instantaneously the highest velocity occurs within the first timestep and gradually decays as the simulation goes on, though there is still a small amount of deformation ongoing after 10,000 years. We can also clearly see that the lateral viscosity variations give rise to assymetrical displacement patterns. This is especially true near the South Pole, where the low viscosity region has enabled the isostatic relaxation to happen much faster than the surrounding regions.

# ![SegmentLocal](displacement_warp.gif "segment")
