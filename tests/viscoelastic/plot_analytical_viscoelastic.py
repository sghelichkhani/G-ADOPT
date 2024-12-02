import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from mpltools import annotation
# resolution/dt testing 14.03.24 on gadi
folder = "./"

# Weerdesteijn resolution/timestep sensitivity figure
fig, axs = plt.subplots(1, 3, figsize=(21, 7))
year_in_seconds = 86400*365.25
tau0 = 23613.45174353904*year_in_seconds  # viscous free surface relaxation time

params = {
    "elastic": {
        "dtf_start": 0.001,
        "nx": 80,
        "xlim":[0.0008, 0.1],
        "sim_time": "short",
        "maxwell": 1e21 / 1e11,
        "annotation_centre": [0.05, 1e-4],
        "subtitle": r"$\Delta$t < $\alpha$"},
    "viscoelastic": {
        "dtf_start": 0.1,
        "nx": 80,
        "xlim":[0.1, 10],
        "sim_time": "long",
        "maxwell": 1e21 / 1e11,
        "annotation_centre": [5, 5e-9],
        "subtitle": r"$\Delta$t ~ $\alpha$"},
    "viscous": {
        "dtf_start": 0.1,
        "nx": 80,
        "xlim":[80, 1e4],
        "sim_time": "long",
        "maxwell": 1e21 / 1e14,
        "annotation_centre": [5000, 5e-9],
        "subtitle": r"$\Delta$t > $\alpha$"},
}

c = 0
for case_name, plot_data in params.items():
    # get correct axes
    axes = axs[c]
    case_name

    # Get list of dts relative to maxwell time
    dtf_start = plot_data["dtf_start"]
    maxwell_time = plot_data["maxwell"]
    dt_factors = dtf_start / (2 ** np.arange(4)) * (tau0/maxwell_time)


    errors = np.loadtxt(f'errors-{case_name}-zhong-free-surface.dat')
    if c==0:
        errors_internal = np.loadtxt(f'errors-elastic-incompressible-internalvariable-coupled-80cells-free-surface.dat')
        marker ='o'
        label='internal variable coupled'
        axes.loglog(dt_factors, errors_internal, marker=marker, color='k', label=label)

        errors_internal_notcoupled = np.loadtxt(f'errors-elastic-incompressible-internalvariable-notcoupled-80cells-free-surface.dat')
        dt_factors_nc = dtf_start / (2 ** np.arange(6)) * (tau0/maxwell_time)
        marker ='o'
        label='internal variable not coupled'
        axes.loglog(dt_factors_nc, errors_internal_notcoupled, ls="--",marker=marker, color='k', label=label)
        

    if c ==1:
        errors_internal = np.loadtxt(f'errors-viscoelastic-incompressible-internalvariable-coupled-320cells-free-surface.dat')
        errors_internal = errors_internal / (2*tau0)
        marker ='o'
        label='internal variable coupled'
        axes.loglog(dt_factors, errors_internal, marker=marker, color='k', label=label)

        errors_internal_notcoupled = np.loadtxt(f'errors-viscoelastic-incompressible-internalvariable-notcoupled-160cells-free-surface.dat')
        errors_internal_notcoupled = errors_internal_notcoupled / (2*tau0)
        dt_factors_nc = dtf_start / (2 ** np.arange(6)) * (tau0/maxwell_time)
        marker ='+'
        label='internal variable not coupled'
        axes.loglog(dt_factors_nc, errors_internal_notcoupled, ls="--", marker=marker, color='k', label=label)

    if c ==2:
        errors_internal = np.loadtxt(f'errors-viscous-incompressible-internalvariable-coupled-80cells-free-surface.dat')
        errors_internal = errors_internal / (2*tau0)
        marker ='o'
        label='internal variable coupled (nz = 80)'
        axes.loglog(dt_factors, errors_internal, marker=marker, color='k', label=label)
        
        errors_internal = np.loadtxt(f'errors-viscous-incompressible-internalvariable-coupled-160cells-free-surface.dat')
        errors_internal = errors_internal / (2*tau0)
        marker ='^'
        label='internal variable coupled (nz = 160)'
        axes.loglog(dt_factors, errors_internal, marker=marker, color='k', label=label)

        errors_internal_notcoupled = np.loadtxt(f'errors-viscous-incompressible-internalvariable-notcoupled-80cells-free-surface.dat')
        errors_internal_notcoupled = errors_internal_notcoupled / (2*tau0)
        marker ='o'
        label='internal variable not coupled'
        axes.loglog(dt_factors, errors_internal_notcoupled, ls="--", marker=marker, color='k', label=label)
        
       # errors_internal_notcoupled = np.loadtxt(f'errors-viscous-incompressible-internalvariable-notcoupled-160cells-free-surface.dat')
       # errors_internal_notcoupled = errors_internal_notcoupled / (2*tau0)
       # dt_factors_nc = dtf_start / (2 ** np.arange(6)) * (tau0/maxwell_time)
       # marker ='v'
       # label='internal variable not coupled (nz =160)'
       # axes.loglog(dt_factors_nc, errors_internal_notcoupled, marker=marker, color='k', label=label)

    if plot_data["sim_time"] == "long":
        # Divide error by simulation time for long experiments 
        errors = errors / (2*tau0)
    # plot peak displacement sensitivity experiments
    label='zhong'
    marker = 'x'
    print(dt_factors)
    print(errors)
    axes.loglog(dt_factors, errors, marker=marker, color='k', label=label)

    # Add slope marker
    annotation.slope_marker(plot_data["annotation_centre"], (1,1), ax=axes, size_frac=0.2)

    # Set axis labels and limits
    x_label = r'$\Delta$t / $\alpha$'
    y_label = 'L2 error'
    fs = 16
    axes.set_xlabel(x_label, fontsize=fs)
    if c == 0:
        axes.set_ylabel(y_label, fontsize=fs)
    
    axes.set_xlim(plot_data['xlim'])
    # Change output from 2^x back to usual numbers
   # axes.xaxis.set_major_formatter(ScalarFormatter())
#    axes.xaxis.set_minor_formatter(ScalarFormatter())
#    axes.yaxis.set_major_formatter(ScalarFormatter())
#    axes.yaxis.set_minor_formatter(ScalarFormatter())


    # Change tick label text size
    axes.tick_params(axis='both', which='major', labelsize=16)
    axes.tick_params(axis='both', which='minor', labelsize=11)
    subtitle = f"{case_name} ("+plot_data["subtitle"]+")"
    axes.set_title(subtitle, fontsize=fs)    
   # axes.xaxis.set_ticks(dt_factors)
    if c ==0:
        # only plot legend on first axis
        axes.legend(fontsize=14)
    c+=1

plt.savefig('02.12.24_zhong_analytical_errors.png')
plt.show()


