import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle

file_list = [["240428-202552", "240428-202159", "240428-203506"],
             ["240428-202348", "240428-201101", "240428-202425"]]
label_list = ["MUPO", "DSAC"]
color_list_1 = [(10/255, 90/255, 100/255),
              (40/255, 150/255, 190/255), 
              (80/255, 180/255, 250/255)]
color_list_2 = [(14/255, 100/255, 10/255), 
              (44/255, 160/255, 44/255), 
              (80/255, 210/255, 84/255)] 
init_pos_list = ["+0.01 [m]", "  0.00 [m]", "- 0.01 [m]"]
font_size = 18
font = {'fontsize': 18}
line_width = 3
plt.rcdefaults()
plt.rcParams['mathtext.default'] = 'regular'


assert len(file_list) == len(label_list) == 2
assert len(file_list[0]) == len(file_list[1]) == len(init_pos_list)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
plt.sca(ax1)

for enum, file in enumerate(file_list[0]):
    order = len(file_list[0]) - enum
    x_encounter = np.loadtxt("../results/figures/MUPO-pyth_encounter/" + file + "/State-1.csv", delimiter=",", skiprows=1)[0][1:]
    y_encounter = np.loadtxt("../results/figures/MUPO-pyth_encounter/" + file + "/State-2.csv", delimiter=",", skiprows=1)[0][1:]
    steer_encounter = np.loadtxt("../results/figures/MUPO-pyth_encounter/" + file + "/Action-1.csv", delimiter=",", skiprows=1)[1:]
    acc_encounter = np.loadtxt("../results/figures/MUPO-pyth_encounter/" + file + "/Action-2.csv", delimiter=",", skiprows=1)[1:]
    # time_step = 0.1s
    time = np.arange(0, len(x_encounter) * 0.1, 0.1)
    linestyle = (0, (3, 3)) if enum == 0 else '-'
    
    plt.plot(time, acc_encounter, label=init_pos_list[enum], color=color_list_1[enum], linewidth=line_width, linestyle=linestyle, zorder=order)

plt.legend(fontsize=font_size, loc='upper right', frameon=False)
plt.tick_params(labelsize=font_size)
plt.xlabel(r"$t \ [s]$", fontsize=font_size)
plt.ylabel(r"$a_x \ [m/s^2]$", fontsize=font_size)
plt.ylim(-2.0, 2.0)
plt.xlim(0, 15)
ax1.set_title(label_list[0], fontsize=font_size)

plt.sca(ax2)
for enum, file in enumerate(file_list[1]):
    order = len(file_list[1]) - enum
    data_x = np.loadtxt("../results/figures/DSAC2-pyth_encounter/" + file + "/State-1.csv", delimiter=",", skiprows=1)[0][1:]
    data_y = np.loadtxt("../results/figures/DSAC2-pyth_encounter/" + file + "/State-2.csv", delimiter=",", skiprows=1)[0][1:]
    data_steer = np.loadtxt("../results/figures/DSAC2-pyth_encounter/" + file + "/Action-1.csv", delimiter=",", skiprows=1)[1:]
    data_acc = np.loadtxt("../results/figures/DSAC2-pyth_encounter/" + file + "/Action-2.csv", delimiter=",", skiprows=1)[1:]
    time = np.arange(0, len(x_encounter) * 0.1, 0.1)

    plt.plot(time, data_acc, label=init_pos_list[enum], color=color_list_2[enum], linewidth=line_width, linestyle='-', zorder=order)
    
plt.legend(fontsize=font_size, frameon=False)
plt.tick_params(labelsize=font_size)
plt.xlabel(r"${t \ [s]}$", fontsize=font_size)
# plt.ylabel(r"$\delta \ [rad]$", fontsize=font_size)

plt.ylim(-2.0, 2.0)
plt.xlim(0, 15)
ax2.set_title(label_list[1], fontsize=font_size)

plt.savefig("../results/figures/encounter_closedloop_action_comparing.pdf", bbox_inches='tight', dpi=300)
plt.close()
######################################################################################################
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 5))
plt.sca(ax1)

rect = Rectangle((13-2/2, -4.8/2), 2, 4.8, facecolor='white', linewidth=2, edgecolor='black', fill=True, hatch='//', alpha=1.0)
plt.gca().add_patch(rect)
for enum, file in enumerate(file_list[0]):
    order = len(file_list[0]) - enum
    data_x = np.loadtxt("../results/figures/MUPO-pyth_encounter/" + file + "/State-1.csv", delimiter=",", skiprows=1)[0][1:]
    data_y = np.loadtxt("../results/figures/MUPO-pyth_encounter/" + file + "/State-2.csv", delimiter=",", skiprows=1)[0][1:]
    data_phi = np.loadtxt("../results/figures/MUPO-pyth_encounter/" + file + "/State-3.csv", delimiter=",", skiprows=1)[0][1:]
    ego_x_5sec = data_x[50]
    ego_y_5sec = data_y[50]
    ego_phi_5sec = data_phi[50]
    # set order
    linestyle = (0, (3, 3)) if enum == 0 else '-'
    ego_vehicle = Rectangle((ego_x_5sec-4.8/2, ego_y_5sec-2.0/2), 4.8, 2.0, angle=ego_phi_5sec*180/np.pi, linewidth=2, edgecolor=color_list_1[enum], fill=False, alpha=1.0, zorder=order, linestyle=linestyle)
    plt.gca().add_patch(ego_vehicle)

plt.tick_params(labelsize=font_size)
# plt.xlabel(r"$p_x \ [m]$", fontsize=font_size)
plt.ylabel(r"$p_y \ [m]$", fontsize=font_size)
plt.ylim(-5, 5)
plt.xlim(0, 30)
ax1.set_aspect('equal', adjustable='box')
ax1.set_title(label_list[0], fontsize=font_size)
# plt.grid()

plt.sca(ax2)

rect = Rectangle((13-2/2, -4.8/2), 2, 4.8, facecolor='white', linewidth=2, edgecolor='black', fill=True, hatch='//', alpha=1.0)
plt.gca().add_patch(rect)
for enum, file in enumerate(file_list[1]):
    order = len(file_list[1]) - enum
    data_x = np.loadtxt("../results/figures/DSAC2-pyth_encounter/" + file + "/State-1.csv", delimiter=",", skiprows=1)[0][1:]
    data_y = np.loadtxt("../results/figures/DSAC2-pyth_encounter/" + file + "/State-2.csv", delimiter=",", skiprows=1)[0][1:]
    data_phi = np.loadtxt("../results/figures/DSAC2-pyth_encounter/" + file + "/State-3.csv", delimiter=",", skiprows=1)[0][1:]
    ego_x_5sec = data_x[50]
    ego_y_5sec = data_y[50]
    ego_phi_5sec = data_phi[50]
    ego_vehicle = Rectangle((ego_x_5sec-4.8/2, ego_y_5sec-2.0/2), 4.8, 2.0, angle=ego_phi_5sec*180/np.pi, linewidth=2, edgecolor=color_list_2[enum], fill=False, alpha=1.0, zorder=order)
    plt.gca().add_patch(ego_vehicle)

plt.tick_params(labelsize=font_size)
plt.xlabel(r"$p_x \ [m]$", fontsize=font_size)
plt.ylabel(r"$p_y \ [m]$", fontsize=font_size)

plt.ylim(-5, 5)
plt.xlim(0, 30)
ax2.set_aspect('equal', adjustable='box')
ax2.set_title(label_list[1], fontsize=font_size)
# plt.grid()

plt.subplots_adjust(hspace=0.5)

plt.savefig("../results/figures/encounter_closedloop_position_comparing.pdf", bbox_inches='tight', dpi=300)
plt.close()