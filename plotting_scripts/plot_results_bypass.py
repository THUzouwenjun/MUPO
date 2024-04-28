import numpy as np
import matplotlib.pyplot as plt

task = "Bypass"
env_id = "pyth_bypass"
algrithm_list = ["MUPO", "DSAC2","SAC"]
title_list = ["MUPO", "DSAC","SAC"]
date_list_list = [["240428-140605", "240428-142943", "240428-145326", "240428-151713", "240428-154045"],
                  ["240427-170601", "240427-172833", "240427-175153", "240427-181352", "240427-183410"],
                  ["240427-150653", "240427-152616", "240427-154536", "240427-160517", "240427-162428"]]

save_dir = "./figures/"
color_list = [(31/255, 119/255, 200/255), 
              (44/255, 160/255, 44/255), 
              (250/255, 120/255, 20/255), 
              (227/255, 119/255, 194/255), 
              (127/255, 127/255, 127/255), 
              (188/255, 189/255, 34/255), 
              (23/255, 190/255, 207/255)]

assert len(algrithm_list) == len(date_list_list) == len(title_list), "The length of algrithm and date_list should be the same."
algrithm_num = len(algrithm_list)

font_size = 18
font = {'fontsize': 18}
line_width = 2
plt.rcdefaults()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.axhline(y=0, color='r', linestyle='--', linewidth=2.0, label="safety limit")
ax1.legend(fontsize=font_size, loc='lower right', frameon=False)
for i in range(algrithm_num):
    # load data
    algrithm_name = algrithm_list[i]
    date_list = date_list_list[i]
    date_num = len(date_list)

    iter_array = None
    con_array_list = []
    tar_array_list = []

    avg_max_return = 0
    for j in range(date_num):
        con_csv_dir = "./results/" + env_id + "/" + algrithm_name + "/" + date_list[j] + "/data/Evaluation_1. CON-RL iter.csv"
        tar_csv_dir = "./results/" + env_id + "/" + algrithm_name + "/" + date_list[j] + "/data/Evaluation_1. TAR-RL iter.csv"
        con_data = np.loadtxt(con_csv_dir, delimiter=",", skiprows=1)
        tar_data = np.loadtxt(tar_csv_dir, delimiter=",", skiprows=1)
        avg_max_return += np.max(tar_data[:, 1])
        if iter_array is None:
            iter_array = con_data[:, 0]
        con_array_list.append(con_data[:, 1])
        tar_array_list.append(tar_data[:, 1])
    avg_max_return = avg_max_return / date_num
    
    print("algrithm: ", title_list[i])
    print("avg_max_return: ", avg_max_return)
    print("====================")

    # plot the average curve
    con_array = np.array(con_array_list)
    tar_array = np.array(tar_array_list)
    con_mean = np.mean(con_array, axis=0)
    con_std = np.std(con_array, axis=0)
    tar_mean = np.mean(tar_array, axis=0)
    tar_std = np.std(tar_array, axis=0)

    # smooth with moving average
    window_size = 3
    std_window_size = 3
    con_mean = np.append(con_mean, np.ones(window_size - 1)*con_mean[-1])
    con_std = np.append(con_std, np.ones(std_window_size - 1)*con_std[-1])
    tar_mean = np.append(tar_mean, np.ones(window_size - 1)*tar_mean[-1])
    tar_std = np.append(tar_std, np.ones(std_window_size - 1)*tar_std[-1])
    con_mean = np.convolve(con_mean, np.ones(window_size)/window_size, mode='valid')
    con_std = np.convolve(con_std, np.ones(std_window_size)/std_window_size, mode='valid')
    tar_mean = np.convolve(tar_mean, np.ones(window_size)/window_size, mode='valid')
    tar_std = np.convolve(tar_std, np.ones(std_window_size)/std_window_size, mode='valid')

    zorder = algrithm_num - i

    ax1.plot(iter_array, con_mean, label=title_list[i], zorder=zorder, color=color_list[i], linewidth=line_width)
    ax1.fill_between(iter_array, con_mean - con_std, con_mean + con_std, zorder=zorder, color=color_list[i], alpha=0.2, edgecolor='none')
    ax1.set_xlabel("Iteration", fontsize=font_size)
    ax1.set_ylabel("Maximum constraint violation", fontsize=font_size)
    # ax1.grid(True)
    ax1.set_xlim(0, 60000)
    ax1.set_ylim(-0.5, 0.5)
    ax1.set_title(task, fontsize=font_size)

    ax2.plot(iter_array, tar_mean, label=title_list[i], zorder=zorder, color=color_list[i], linewidth=line_width)
    ax2.fill_between(iter_array, tar_mean - tar_std, tar_mean + tar_std, alpha=0.2, zorder=zorder, color=color_list[i], edgecolor='none')
    ax2.set_xlabel("Iteration", fontsize=font_size)
    ax2.set_ylabel("Average episode return", fontsize=font_size)
    # ax2.grid(True)
    ax2.set_xlim(0, 60000)
    ax2.set_ylim(50, 130)

    ax2.legend(fontsize=font_size, loc='lower right', frameon=False)
    ax2.set_title(task, fontsize=font_size)

plt.savefig(save_dir + "/" + task + "_combined.pdf", bbox_inches='tight', dpi=300)

plt.show()