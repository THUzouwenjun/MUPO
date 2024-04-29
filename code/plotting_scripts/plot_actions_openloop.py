import matplotlib.pyplot as plt
import numpy as np
from gops.utils.plot_q import plt_action_disconti

file_list = ["240428-164508", "240428-171214", "240428-171415"]
label_list = ["MUPO", "DSAC", "SAC"]
alg_list = ["MUPO", "DSAC2", "SAC"]
color_list = [(31/255, 119/255, 200/255), 
              (44/255, 160/255, 44/255), 
              (250/255, 120/255, 20/255), 
              (227/255, 119/255, 194/255), 
              (127/255, 127/255, 127/255), 
              (188/255, 189/255, 34/255), 
              (23/255, 190/255, 207/255)]
font_size = 18
font = {'fontsize': 18}
line_width = 3
plt.rcdefaults()
plt.rcParams['mathtext.default'] = 'regular'

# plot bypass
for enum, file in enumerate(file_list):
    order = len(file_list) - enum
    alg_path = "../results/figures/" + alg_list[enum] + "-pyth_bypass/"
    x_bypass = np.loadtxt(alg_path + file + "/State-1.csv", delimiter=",", skiprows=1)[0][1:]
    y_bypass = np.loadtxt(alg_path + file + "/State-2.csv", delimiter=",", skiprows=1)[0][1:]
    steer_bypass = np.loadtxt(alg_path + file + "/Action-1.csv", delimiter=",", skiprows=1)[1:]
    acc_bypass = np.loadtxt(alg_path + file + "/Action-2.csv", delimiter=",", skiprows=1)[1:]

    plt_action_disconti(y_bypass, steer_bypass, label=label_list[enum], color=color_list[enum], line_width=line_width, order=order)


plt.legend(fontsize=font_size, frameon=False)
plt.tick_params(labelsize=font_size)

plt.xlabel(r"$p_y \ [m]$", fontsize=font_size)
plt.ylabel(r"$\delta \ [rad]$", fontsize=font_size)
plt.ylim(-0.5, 0.5)
plt.yticks(np.arange(-0.4, 0.5, 0.4))
plt.xlim(-2, 2)
# plt.grid()

plt.savefig("../results/figures/bypass_steer.pdf", bbox_inches='tight', dpi=300)
plt.close()

# plot encounter
file_list = ["240428-164915", "240428-171819", "240428-171859"]
label_list = ["MUPO", "DSAC", "SAC"]
alg_list = ["MUPO", "DSAC2", "SAC"]

for enum, file in enumerate(file_list):
    order = len(file_list) - enum
    alg_path = "../results/figures/" + alg_list[enum] + "-pyth_encounter/"
    data_x = np.loadtxt(alg_path + file + "/State-1.csv", delimiter=",", skiprows=1)[0][1:]
    data_y = np.loadtxt(alg_path + file + "/State-2.csv", delimiter=",", skiprows=1)[0][1:]
    date_steer = np.loadtxt(alg_path + file + "/Action-1.csv", delimiter=",", skiprows=1)[1:]
    date_acc = np.loadtxt(alg_path + file + "/Action-2.csv", delimiter=",", skiprows=1)[1:]
    plt_action_disconti(data_x, date_acc, label=label_list[enum], color=color_list[enum], threshold=1.5, line_width=line_width, order=order)
plt.legend(fontsize=font_size, frameon=False)
plt.tick_params(labelsize=font_size)

plt.xlabel(r"${p_x \ [m]}$", fontsize=font_size)
plt.ylabel(r"${a_x \ [m/s^2]}$", fontsize=font_size)

plt.ylim(-2.0, 2.0)
plt.xlim(-3, 3)
# plt.grid()
plt.savefig("../results/figures/encounter_acc.pdf", bbox_inches='tight', dpi=300)
plt.close()
