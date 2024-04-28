#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: iDLab
#  Description: Evaluation of trained policy
#  Update Date: 2021-05-10, Yang Guan: renew environment parameters


import numpy as np
import torch

from gops.create_pkg.create_env import create_env
from gops.utils.common_utils import set_seed

class Evaluator:
    def __init__(self, index=0, **kwargs):
        kwargs.update({
            "reward_scale": None,
            "repeat_num": None,
            "gym2gymnasium": False,
            "vector_env_num": None,
            "is_eval": True,
        })
        self.env = create_env(**kwargs)

        _, self.env = set_seed(kwargs["trainer"], kwargs["seed"], index + 400, self.env)

        alg_name = kwargs["algorithm"]
        alg_file_name = alg_name.lower()
        file = __import__(alg_file_name)
        ApproxContainer = getattr(file, "ApproxContainer")
        kwargs["device"] = "cpu"
        self.networks = ApproxContainer(**kwargs)
        self.render = kwargs["is_render"]

        self.num_eval_episode = kwargs["num_eval_episode"]
        self.action_type = kwargs["action_type"]
        self.policy_func_name = kwargs["policy_func_name"]
        self.save_folder = kwargs["save_folder"]
        self.eval_save = kwargs.get("eval_save", True)

        self.print_time = 0
        self.print_iteration = -1

    def load_state_dict(self, state_dict):
        self.networks.load_state_dict(state_dict)

    def run_an_episode(self, iteration, index, render=True, init_state=None):
        if self.print_iteration != iteration:
            self.print_iteration = iteration
            self.print_time = 0
        else:
            self.print_time += 1
        obs_list = []
        action_list = []
        reward_list = []
        if init_state is None:
            obs, info = self.env.reset()
        else:
            obs, info = self.env.reset(init_state=init_state)
        done = 0
        info["TimeLimit.truncated"] = False
        max_constraint = -10.0
        while not (done or info["TimeLimit.truncated"]):
            batch_obs = torch.from_numpy(np.expand_dims(obs, axis=0).astype("float32"))
            action = self.networks.compute_action(batch_obs, deterministic=True)
            action = action.cpu().detach().numpy()[0]
            next_obs, reward, done, next_info = self.env.step(action)
            obs_list.append(obs)
            action_list.append(action)
            obs = next_obs
            info = next_info
            if "TimeLimit.truncated" not in info.keys():
                info["TimeLimit.truncated"] = False
            if "constraint" in info.keys():
                if (info["constraint"][0] > max_constraint):
                    max_constraint = info["constraint"][0]
                # print("max_constraint", max_constraint)
            # Draw environment animation
            if render:
                self.env.render()
            reward_list.append(reward)
        eval_dict = {
            "reward_list": reward_list,
            "action_list": action_list,
            "obs_list": obs_list,
        }
        if self.eval_save:
            np.save(
                self.save_folder
                + "/evaluator/iter{}_ep{}".format(iteration, self.print_time),
                eval_dict,
            )
        episode_return = sum(reward_list)
        return episode_return, max_constraint

    def run_n_episodes(self, n, iteration):
        episode_return_list = []
        episode_constraint_list = []
        for index in range(n):
            # select n initial states uniformly from the initial state of env
            if hasattr(self.env, "work_space"):
                work_space = self.env.work_space
                lb = work_space[0]
                ub = work_space[1]
                init_state = index/float(n-1) * ub + (1-index/float(n-1)) * lb
                episode_return, max_constraint = self.run_an_episode(iteration, index, self.render, init_state) 
            else:
                episode_return, max_constraint = self.run_an_episode(iteration, index, self.render) 
            episode_return_list.append(episode_return)
            episode_constraint_list.append(max_constraint)
        return np.mean(episode_return_list), np.max(episode_constraint_list)

    def run_evaluation(self, iteration):
        return self.run_n_episodes(self.num_eval_episode, iteration)
