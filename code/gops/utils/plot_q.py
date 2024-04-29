#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: Plot Q Function
#  Update: 2023-04-12, Wenjun Zou: Revise Codes


import os
import matplotlib.pyplot as plt
import torch
import argparse
import numpy as np
from gops.utils.common_utils import get_args_from_json, mp4togif
import matplotlib.patches as mpatches

def load_args(log_policy_dir: str):
    json_path = os.path.join(log_policy_dir, "config.json")
    parser = argparse.ArgumentParser()
    args_dict = vars(parser.parse_args())
    args = get_args_from_json(json_path, args_dict)
    return args

def load_policy(log_policy_dir: str, trained_policy_iteration: str, args):
    # Create policy
    alg_name = args["algorithm"]
    alg_file_name = alg_name.lower()
    file = __import__(alg_file_name)
    ApproxContainer = getattr(file, "ApproxContainer")
    networks = ApproxContainer(**args)

    # Load trained policy
    log_path = log_policy_dir + "/apprfunc/apprfunc_{}.pkl".format(
        trained_policy_iteration
    )
    networks.load_state_dict(torch.load(log_path))
    return networks


def plot_q_function_3D(state, networks, save_path=None):

    action_dim = networks.q_args["act_dim"]
    assert action_dim == 2, print("Only support 2D action space")
    act_high_lim = networks.q_args["act_high_lim"]
    act_low_lim = networks.q_args["act_low_lim"]

    action_num_per_dim = 11
    action1 = np.linspace(act_low_lim[0], act_high_lim[0], action_num_per_dim)
    action2 = np.linspace(act_low_lim[1], act_high_lim[1], action_num_per_dim)
    action1, action2 = np.meshgrid(action1, action2)
    action1_flat = action1.reshape(-1)
    action2_flat = action2.reshape(-1)
    action = np.stack([action1_flat, action2_flat], axis=1)
    action = torch.from_numpy(action.astype("float32"))

    # Get the Q function
    action_num = action.shape[0]
    batch_obs = torch.from_numpy(np.expand_dims(state, axis=0).astype("float32"))
    batch_obs_repeat = batch_obs.repeat(action_num, 1)
    q = networks.compute_q_values(batch_obs_repeat, action).cpu().detach().numpy()
    q = q.reshape(action_num_per_dim, action_num_per_dim)

    action_pi = networks.compute_action(batch_obs, deterministic=True).detach()
    batch_obs = batch_obs.cpu().type(torch.float32)
    action_pi = action_pi.cpu().type(torch.float32)
    q_pi = networks.compute_q_values(batch_obs, action_pi).cpu().detach().numpy()

    plt.figure()
    ax3 = plt.axes(projection='3d')
    ax3.plot_surface(action1, action2, q, rstride=1, cstride=1, cmap='rainbow')
    ax3.set_xlabel('steer')
    ax3.set_ylabel('acc')
    ax3.set_zlabel('Q')
    ax3.invert_xaxis()

    ax3.scatter(action_pi[0, 0], action_pi[0, 1], q_pi + 0.1, c='k', marker='o', s=300, zorder=10)

    plt.savefig(save_path)
    plt.close()


def plot_pi_2D(state, networks, save_path=None):
    action_dim = networks.q_args["act_dim"]
    assert action_dim == 2, print("Only support 2D action space")
    act_high_lim = networks.q_args["act_high_lim"]
    act_low_lim = networks.q_args["act_low_lim"]
    sub_policy_num = networks.sub_policy_num

    state = torch.from_numpy(np.expand_dims(state, axis=0).astype("float32"))
    dist = networks.create_action_distributions(state)
    act_mode = dist.mode()[0].cpu().detach().numpy()
    plt.figure()
    mean_with_max_prob = np.zeros((sub_policy_num, 2))
    std_with_max_prob = np.zeros((sub_policy_num, 2))
    max_prob = 0.0
    for i in range(sub_policy_num):
        mean, std, gate_prob = dist.get_sub_mean_std(i)
        mean = mean.cpu().detach().numpy()[0]
        std = std.cpu().detach().numpy()[0]
        gate_prob = gate_prob.cpu().detach().numpy()[0]
        if gate_prob > max_prob:
            max_prob = gate_prob
            mean_with_max_prob = mean
            std_with_max_prob = std
        r = max(0, 2*(gate_prob - 0.5))
        g = max(0, 2*(0.5 - abs(gate_prob - 0.5)))
        b = max(0, 2*(0.5 - gate_prob))

        color = (r, g, b)
        ellipse = mpatches.Ellipse(mean, std[0], std[1], color=color)
        alpha = 0.05 + 0.95 * gate_prob
        ellipse.set_alpha(alpha)
        plt.gca().add_patch(ellipse)

    ellipse = mpatches.Ellipse(mean_with_max_prob, std_with_max_prob[0], std_with_max_prob[1], edgecolor='r', facecolor='none')
    plt.gca().add_patch(ellipse)
    plt.scatter(act_mode[0], act_mode[1], c='b', marker='o', s=50, zorder=10)
    # 作图
    plt.xlim(act_low_lim[0], act_high_lim[0])
    plt.ylim(act_low_lim[1], act_high_lim[1])
    plt.xlabel('steer')
    plt.ylabel('acc')

    plt.savefig(save_path)
    plt.close()
    
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

import os   

class VideoCreator:
    def __init__(self, save_path):
        self.save_path = save_path

    def create_video(self):
        combined_path = os.path.join(self.save_path, "combined")
        if not os.path.exists(combined_path):
            os.makedirs(combined_path)

        num_images = len(os.listdir(os.path.join(self.save_path, "q")))

        for i in range(num_images):
            q_image_path = os.path.join(self.save_path, "q", f"{i}.png")
            pi_image_path = os.path.join(self.save_path, "pi", f"{i}.png")
            output_image_path = os.path.join(combined_path, f"{i}.png")
            os.system(f"ffmpeg -i {q_image_path} -i {pi_image_path} -filter_complex hstack {output_image_path} > /dev/null 2>&1")

        os.system(f"ffmpeg -r 30 -i {combined_path}/%d.png -vcodec libx264 -pix_fmt yuv420p {self.save_path}/videos/combined.mp4")
