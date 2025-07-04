import csv
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from PIL import Image
import numpy as np
import os


plt.rc('font', family='Times New Roman')


def read_csv(filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
    return data


def prepare_data(filename, max_time=None, interval_scale=1):
    data = read_csv(filename)
    # create dataframe from data, use the firt row as column header
    df = pd.DataFrame(data, columns=data[0])
    # drop the first row
    df = df.drop(0)
    if max_time is not None:
        df['step'] = pd.to_numeric(df['step'])
        df = df[df['step'] * 0.02 <= max_time]

    states = np.array([df['ego_x'], df['ego_y'], df['ego_theta'], df['ego_v'], df['ego_w'],
                df['error_lat'], df['error_theta'], df['error_v'],
                df['surro_x'], df['surro_y'], df['surro_theta'], df['surro_v'], df['surro_w']], dtype=np.float32).T
    t = np.arange(0, len(states), 1) * 0.02

    # Apply moving average with gradually increasing window size
    window_size = 50
    for i in [0, 1]:  # assuming 0 is x and 1 is y
        states_df = pd.Series(states[:, i])
        rolling_series = states_df.rolling(window=window_size, min_periods=1).mean()
        states[:, i] = rolling_series.values

    states = states[::interval_scale, :]
    t = t[::interval_scale]
    return states, t

def plot_trajectory_list(states_list, t_list, save_path=None, title=None):
    fig, ax = plt.subplots(figsize=(9, 9.5))
    ax.set_aspect('equal')
    ax.set_ylim(-5, 5)
    ax.set_xlim(-4, 4)
    # ax.set_title('Trajectories', fontsize=20)
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)

    # grid on
    ax.grid(True)
    
    # Plot the reference trajectory
    y1 = np.linspace(-6, 6, 1000)
    x1 = np.zeros_like(y1)
    ax.plot(x1, y1, "k", zorder=11)

    for states, t in zip(states_list, t_list):
        ego_x, ego_y, ego_theta, ego_v, ego_w, error_lat, error_theta, error_v, obs_x, obs_y, obs_theta, obs_v, obs_w = states.T
        
        # Plot initial ego and obstacle positions
        circle_ego_init = Circle((ego_x[0], ego_y[0]), radius=0.74/2, edgecolor='indigo', fill=False, alpha=0.2, zorder=10)
        ax.add_patch(circle_ego_init)
        
        # Calculate bottom-left corner of the square for the obstacle
        bottom_left_x = obs_x[0] - 0.5
        bottom_left_y = obs_y[0] - 0.5
        square_obs_init = Rectangle((bottom_left_x, bottom_left_y), 1.0, 1.0, edgecolor='green', fill=False, alpha=0.2, zorder=10)
        ax.add_patch(square_obs_init)

        # Plot ego positions at different times
        for i in range(len(t)):
            sc1 = ax.scatter(ego_x[i], ego_y[i], c=[t[i]], cmap='inferno_r', vmin=0, vmax=25, zorder=10)

        idx = np.argmax(t >= 5)
        circle_ego_mid = Circle((ego_x[idx], ego_y[idx]), radius=0.74/2, edgecolor='indigo', fill=False, alpha=0.6, zorder=10)
        ax.add_patch(circle_ego_mid)

    cbar1 = fig.colorbar(sc1, fraction=0.06)
    cbar1.set_label('Time (s)', fontsize=22)
    cbar1.ax.tick_params(labelsize=20)

    ego_handle = ax.plot([],[], 'o', color='indigo', label='Ego')[0] 
    surr_handle = ax.plot([],[], 'o', color='green', label='Obstacle')[0]
    ax.legend(handles=[ego_handle, surr_handle], fontsize=20, loc='upper left')

    if title is not None:
        plt.title(title, fontsize=22)

    # plt.pause(1)
    if save_path is not None:
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close()

def find_all_png(read_path):
    #find all png in the folder, search only the directory of this level, and do not search for subdirectories
    file_list = []
    for file in os.listdir(read_path):
        if file.endswith('.png') and not file.endswith('compressed.png'):
            file_list.append(os.path.join(read_path, file))
    return file_list

def compress_png(read_path, save_path=None, size_scale=0.9, quality=30):
    if save_path is None:
        save_path = os.path.dirname(read_path) + '/compressed_png'
        os.makedirs(save_path, exist_ok=True)
        save_path = save_path + '/' + os.path.basename(read_path)
    img = Image.open(read_path)
    # Resize
    width, height = img.size
    img = img.resize((int(width*size_scale), int(height*size_scale)))
    img.save(save_path, 
            optimize=True, 
            quality=quality) # 0 to 100, lower is more compression

if __name__ == "__main__":
    read_path_list = [
        ["../data/results/ugv_results/11-09-16-18-48", None],
        ["../data/results/ugv_results/11-09-16-10-45", None],
        ["../data/results/ugv_results/11-09-16-17-46", None],
        ["../data/results/ugv_results/11-09-16-31-34", None],
    ]
    states_list = []
    t_list = []
    for read_path, max_time in read_path_list:
        # 调用prepare_data函数，读取csv文件，返回状态和时间
        states, t = prepare_data(read_path + '/results0.csv', max_time=max_time, interval_scale=2)
        states_list.append(states)
        t_list.append(t)

    plot_trajectory_list(states_list, t_list, save_path='../results/figures/UGV_bifur.pdf', title="bifurcated policy")

    read_path_list = [
        ["../data/results/ugv_results/11-09-16-20-08_conti", None],
        ["../data/results/ugv_results/11-09-16-34-10_conti", None],
        ["../data/results/ugv_results/11-09-16-37-53_conti", None],
        ["../data/results/ugv_results/11-09-16-45-18_conti", None],
    ]
    states_list = []
    t_list = []
    for read_path, max_time in read_path_list:
        # 调用prepare_data函数，读取csv文件，返回状态和时间
        states, t = prepare_data(read_path + '/results0.csv', max_time=max_time, interval_scale=2)
        states_list.append(states)
        t_list.append(t)

    plot_trajectory_list(states_list, t_list, save_path='../results/figures/UGV_conti.pdf', title="continuous policy")
