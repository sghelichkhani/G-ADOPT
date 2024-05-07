from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def obtain_rol_outputs(path_to_file):
    rol_outputs = []
    with open(file=path_to_file, mode="r") as fi:
        for line in fi:
            try:
                line_as_array = np.asarray(line.strip().split()).astype("float")
                if line_as_array.size > 0:
                    rol_outputs.append(line_as_array)
            except ValueError:
                pass
    return np.asarray(rol_outputs)


def obtain_callback_outputs(path_to_file):
    with open(file=path_to_file, mode="r") as fi:
        lines = fi.readlines()
    lines = [line.replace(";", "") for line in lines]
    flgs = ["Initial misfit" in line for line in lines]
    initial_misfits = np.asarray([line.strip().split()[2] for line in np.asarray(lines)[flgs]]).astype("float")
    final_misfits = np.asarray([line.strip().split()[5] for line in np.asarray(lines)[flgs]]).astype("float")
    return initial_misfits, final_misfits


base_path = Path(".")
all_files = base_path.rglob("job_WEIGHTU_0.1_WAVELENGTH_*/output.log")

all_outputs = {}
for file_name in list(all_files):
    rol_output = obtain_rol_outputs(file_name)
    init_misfit, final_misfit = obtain_callback_outputs(path_to_file=file_name)
    all_outputs[file_name.parts[0]] = {"rol": rol_output, "init_misfit": init_misfit, "final_misfit": final_misfit}


# Creating subplots
plt.close(1)
fig, axs = plt.subplots(1, 3, figsize=(15, 5), num=1)

for it, (name, simu) in enumerate(all_outputs.items()):
    for id, (ax, array_to_plot) in enumerate(zip(axs, [simu["init_misfit"], simu["final_misfit"], simu["rol"][:, 1]])):
        # Initial Misfit plot
        ax.plot(array_to_plot, color=cm.get_cmap("jet", len(all_outputs))(it), label=f"{name}")
        ax.set_yscale('log')
        ax.set_xlabel('Iteration')
fig.tight_layout()

for ax in axs:
    ax.grid(True)
axs[-1].legend()

fig.show()
