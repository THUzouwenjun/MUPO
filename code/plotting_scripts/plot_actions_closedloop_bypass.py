import matplotlib.pyplot as plt
import numpy as np
# from gops.utils.plot_q import plt_action_disconti
from matplotlib.patches import FancyArrowPatch, Rectangle

def plt_action_disconti(step_array, action_array, label, color, threshold=0.4, line_width=2.0, order=1, linestyle = '-'):
    discontinuity_index = []
    discontinuity_index.append(0)
    for i in range(len(action_array)-1):
        if abs(action_array[i] - action_array[i+1]) > threshold:
            # print("discontinuity at step: ", step_array[i])
            discontinuity_index.append(i+1)
    discontinuity_index.append(len(action_array)-1)
    
    for i in range(len(discontinuity_index)-1):
        start_index = discontinuity_index[i]
        end_index = discontinuity_index[i+1]
        if i == 0:
            plt.plot(step_array[start_index:end_index], action_array[start_index:end_index], label=label, color=color, linewidth=line_width, zorder=order, linestyle=linestyle)
        else:
            plt.plot(step_array[start_index:end_index], action_array[start_index:end_index], color=color, linewidth=line_width, zorder=order, linestyle=linestyle)

file_list = [["240428-174721", "240428-174642", "240428-174614"], 
             ["240428-173729", "240428-173605", "240428-173845"]]
label_list = ["MUPO", "DSAC"]
color_list_1 = [(10/255, 90/255, 100/255), 
              (40/255, 150/255, 190/255), 
              (80/255, 180/255, 250/255)] 
color_list_2 = [(14/255, 100/255, 10/255), 
              (44/255, 160/255, 44/255), 
              (80/255, 210/255, 84/255)] 
init_pos_list = [r'$p_{y0} = 0.01~m$', r'$p_{y0} = 0.00~m$', r'$p_{y0} = -0.01~m$']
font_size = 18
font = {'fontsize': 18}
line_width = 3
plt.rcdefaults()
plt.rcParams['mathtext.default'] = 'regular'


assert len(file_list) == len(label_list) == 2
assert len(file_list[0]) == len(file_list[1]) == len(init_pos_list)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
plt.sca(ax1)

load_dir = "../data/results"

for enum, file in enumerate(file_list[0]):

    order = len(file_list[0]) - enum
    x_bypass = np.loadtxt(load_dir + "/figures/MUPO-pyth_bypass/" + file + "/State-1.csv", delimiter=",", skiprows=1)[0][1:]
    y_bypass = np.loadtxt(load_dir + "/figures/MUPO-pyth_bypass/" + file + "/State-2.csv", delimiter=",", skiprows=1)[0][1:]
    steer_bypass = np.loadtxt(load_dir + "/figures/MUPO-pyth_bypass/" + file + "/Action-1.csv", delimiter=",", skiprows=1)[1:]
    acc_bypass = np.loadtxt(load_dir + "/figures/MUPO-pyth_bypass/" + file + "/Action-2.csv", delimiter=",", skiprows=1)[1:]
    # time_step = 0.1s
    time = np.arange(0, len(x_bypass) * 0.1, 0.1)
    linestyle = (0, (3, 3)) if enum == 0 else '-'
    plt_action_disconti(time, steer_bypass, label=init_pos_list[enum], color=color_list_1[enum], line_width=line_width, order=order, linestyle=linestyle)

plt.legend(fontsize=font_size-2, loc='upper right', frameon=False)
plt.tick_params(labelsize=font_size)
plt.xlabel(r"$t \ [s]$", fontsize=font_size)
plt.ylabel(r"$\delta \ [rad]$", fontsize=font_size)
plt.ylim(-0.5, 0.5)
plt.xlim(0, 20)
ax1.set_title(label_list[0], fontsize=font_size)


plt.sca(ax2)
for enum, file in enumerate(file_list[1]):
    order = len(file_list[1]) - enum
    data_x = np.loadtxt(load_dir + "/figures/DSAC2-pyth_bypass/" + file + "/State-1.csv", delimiter=",", skiprows=1)[0][1:]
    data_y = np.loadtxt(load_dir + "/figures/DSAC2-pyth_bypass/" + file + "/State-2.csv", delimiter=",", skiprows=1)[0][1:]
    data_steer = np.loadtxt(load_dir + "/figures/DSAC2-pyth_bypass/" + file + "/Action-1.csv", delimiter=",", skiprows=1)[1:]
    data_acc = np.loadtxt(load_dir + "/figures/DSAC2-pyth_bypass/" + file + "/Action-2.csv", delimiter=",", skiprows=1)[1:]
    time = np.arange(0, len(x_bypass) * 0.1, 0.1)
    plt_action_disconti(time, data_steer, label=init_pos_list[enum], color=color_list_2[enum], threshold=0.7, line_width=line_width, order=order)
    
plt.legend(fontsize=font_size-2, frameon=False)
plt.tick_params(labelsize=font_size)
plt.xlabel(r"${t \ [s]}$", fontsize=font_size)
# plt.ylabel(r"$\delta \ [rad]$", fontsize=font_size)

plt.ylim(-0.5, 0.5)
plt.xlim(0, 20)
# 标题为DSAC
ax2.set_title(label_list[1], fontsize=font_size)

plt.savefig("../results/figures/bypass_closedloop_action_comparing.pdf", bbox_inches='tight', dpi=300)
plt.close()
######################################################################################################
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 5))
plt.sca(ax1)

rect = Rectangle((10-4.8/2, -1), 4.8, 2, facecolor='white', linewidth=2, edgecolor='black', fill=True, hatch='//', alpha=1.0)
plt.gca().add_patch(rect)
for enum, file in enumerate(file_list[0]):
    order = len(file_list[0]) - enum
    data_x = np.loadtxt(load_dir + "/figures/MUPO-pyth_bypass/" + file + "/State-1.csv", delimiter=",", skiprows=1)[0][1:]
    data_y = np.loadtxt(load_dir + "/figures/MUPO-pyth_bypass/" + file + "/State-2.csv", delimiter=",", skiprows=1)[0][1:]
    linestyle = (0, (3, 3)) if enum == 0 else '-'
    plt_action_disconti(data_x, data_y, label=init_pos_list[enum], color=color_list_1[enum], line_width=line_width, order=order, linestyle=linestyle)

plt.tick_params(labelsize=font_size)
# plt.xlabel(r"$p_x \ [m]$", fontsize=font_size)
plt.ylabel(r"$p_y \ [m]$", fontsize=font_size)
plt.ylim(-5, 5)
plt.xlim(0, 60)
ax1.set_aspect('equal', adjustable='box')
ax1.set_title(label_list[0], fontsize=font_size)
# plt.grid()

plt.sca(ax2)
rect = Rectangle((10-4.8/2, -1), 4.8, 2, facecolor='white', linewidth=2, edgecolor='black', fill=True, hatch='//', alpha=1.0)
plt.gca().add_patch(rect)
for enum, file in enumerate(file_list[1]):
    order = len(file_list[1]) - enum
    data_x = np.loadtxt(load_dir + "/figures/DSAC2-pyth_bypass/" + file + "/State-1.csv", delimiter=",", skiprows=1)[0][1:]
    data_y = np.loadtxt(load_dir + "/figures/DSAC2-pyth_bypass/" + file + "/State-2.csv", delimiter=",", skiprows=1)[0][1:]
    plt_action_disconti(data_x, data_y, label=init_pos_list[enum], color=color_list_2[enum], threshold=0.7, line_width=line_width, order=order)

plt.tick_params(labelsize=font_size)
plt.xlabel(r"$p_x \ [m]$", fontsize=font_size)
plt.ylabel(r"$p_y \ [m]$", fontsize=font_size)

plt.ylim(-5, 5)
plt.xlim(0, 60)
ax2.set_aspect('equal', adjustable='box')
ax2.set_title(label_list[1], fontsize=font_size)
# plt.grid()

plt.subplots_adjust(hspace=0.5)

plt.savefig("../results/figures/bypass_closedloop_position_comparing.pdf", bbox_inches='tight', dpi=300)
plt.close()