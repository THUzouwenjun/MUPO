import importlib
import time
from gops.create_pkg.create_env import create_env
from gops.create_pkg.create_env_model import create_env_model
from gops.sys_simulator.sys_opt_controller import OptController
from gops.utils.common_utils import get_args_from_json
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import argparse
import imageio

# save path
save_path = "/home/zhengzhilong/code/gops/gops/sys_simulator/mpc_logs/" + time.strftime('%Y-%m-%d_%H-%M-%S')
os.makedirs(save_path, exist_ok=True)

def load_args(log_policy_dir):
    json_path = os.path.join(log_policy_dir, "config.json")
    parser = argparse.ArgumentParser()
    args_dict = vars(parser.parse_args())
    args = get_args_from_json(json_path, args_dict)
    return args

def load_policy(log_policy_dir, trained_policy_iteration):
    # Create policy
    args = load_args(log_policy_dir)
    module = importlib.import_module("gops.algorithm." + args['algorithm'].lower())
    networks = getattr(module, "ApproxContainer")(**args)

    # Load trained policy
    log_path = log_policy_dir + "/apprfunc/apprfunc_{}.pkl".format(trained_policy_iteration)
    networks.load_state_dict(torch.load(log_path))
    return networks

algorithm = 'SAC'
env_id = 'idp'
log_policy_dir = os.path.join("code/gops/results/", algorithm, env_id)
if env_id == 's2a1':
    ckpt = '350000'
    init_info = {"init_state":[0.5, -0.5]}
elif env_id == 's3a1':
    ckpt = '155000'
    init_info = {"init_state":[0.1, 0.1, 0.1]}
elif env_id == 's5a1':
    ckpt = '430000'
    init_info = {"init_state":[0.1, 0.2, 0, 0.1, 0]}
elif env_id == 's4a2':
    ckpt = '100000'
    init_info = {"init_state":[0.5, 0.2, 0.5, 0.2]}
elif env_id == 's6a3':
    ckpt = '82500_opt'
    init_info = {"init_state":[0.05, 0.1, 0, 0, 0, 0.1]}

# value_net = load_policy(log_policy_dir, ckpt).v

# def terminal_cost(obs):    
#     return -value_net(obs)

def run_an_episode(env, controller, sim_num, ctrl_interval, init_info={}):
    state_list = []
    action_list = []
    reward_list = []
    obs_list = []
    step = 0
    step_list = []
    # filenames = []
    
    env.seed(seed)
    obs, _ = env.reset(**init_info)
    # env.render()
    state = env.state

    sim_horizon = np.arange(sim_num)
    for i in sim_horizon:
        if (i % ctrl_interval) == 0:
            action = controller(obs)
        state_list.append(state)
        obs_list.append(obs)
        next_obs, reward, _, _ = env.step(action)
        # env.render()
        # plt.savefig(os.path.join(save_path, f"{i}.png"))
        # filenames.append(f"{i}.png")

        action_list.append(action)
        step_list.append(step)
        reward_list.append(reward)

        obs = next_obs
        state = env.state
        step = step + 1
        print("step:", step)

    eval_dict = {
        "reward_list": reward_list,
        "action_list": action_list,
        "state_list": state_list,
        "step_list": step_list,
        "obs_list": obs_list,
    }
                    
    # with imageio.get_writer(os.path.join(save_path, f"render.gif"), mode='I') as gif_writer:
    #     for filename in filenames:
    #         image = imageio.imread(os.path.join(save_path, filename))
    #         gif_writer.append_data(image)
    #         os.remove(os.path.join(save_path, filename))
    return eval_dict

args = load_args(log_policy_dir)
env = create_env(**args)
env.set_mode('test')
model = create_env_model(**args)
action_dim = model.action_dim
state_dim = model.obs_dim
seed = 50
# init_info = {"init_state": np.array([0, -0.2, -0., -0., -0., -0.], dtype=np.float64)}
init_info = {}
sim_num = 1000

legend_list = []
ctrl_interval_list = []
opt_controllers = []

# OPT
# legend_list.append('OPT')
# ctrl_interval_list.append(1)
# opt_controllers.append(env.control_policy)

# MPC w/o TC
num_pred_step = 150
ctrl_interval = 1
opt_args={
    "num_pred_step": num_pred_step, 
    "gamma": 0.99,
    "verbose": 1,
    "ctrl_interval": ctrl_interval,
    "minimize_options": {
        "max_iter": 200, 
        "tol": 1e-3,
        "acceptable_tol": 1e-0,
        "acceptable_iter": 10,
        # "print_level": 5,
        # "print_timing_statistics": "yes",
    },
    "mode": "collocation"
}
legend_list.append(f'MPC-{num_pred_step}-{ctrl_interval} w/o TC')
ctrl_interval_list.append(ctrl_interval)
opt_controllers.append(OptController(model, **opt_args))

# MPC w/o TC
num_pred_step = 175
ctrl_interval = 1
opt_args={
    "num_pred_step": num_pred_step, 
    "gamma": 0.99,
    "ctrl_interval": ctrl_interval,
    "minimize_options": {
        "max_iter": 200, 
        "tol": 1e-3,
        "acceptable_tol": 1e-0,
        "acceptable_iter": 10,
        # "print_level": 5,
    },
    "mode": "collocation",
    # "use_terminal_cost": True,
    # "terminal_cost": terminal_cost,
}
legend_list.append(f'MPC-{num_pred_step}-{ctrl_interval} w/o TC')
ctrl_interval_list.append(ctrl_interval)
opt_controllers.append(OptController(model, **opt_args))

# MPC w/o TC
num_pred_step = 200
ctrl_interval = 1
opt_args={
    "num_pred_step": num_pred_step, 
    "gamma": 0.99,
    "ctrl_interval": ctrl_interval,
    "minimize_options": {
        "max_iter": 200, 
        "tol": 1e-3,
        "acceptable_tol": 1e-0,
        "acceptable_iter": 10,
        # "print_level": 5,
    },
    "mode": "collocation",
    # "use_terminal_cost": True,
}
legend_list.append(f'MPC-{num_pred_step}-{ctrl_interval} w/o TC')
ctrl_interval_list.append(ctrl_interval)
opt_controllers.append(OptController(model, **opt_args))

eval_dict_list = []
time_list = []
for i in range(len(opt_controllers)):
    t1 = time.time()
    eval_dict = run_an_episode(env, opt_controllers[i], sim_num, ctrl_interval_list[i], init_info)
    t2 = time.time()
    eval_dict_list.append(eval_dict)
    time_list.append(t2 - t1)

# plot action
for j in range(action_dim):
    plt.figure()
    for i in range(len(opt_controllers)):
        steps = eval_dict_list[i]['step_list']
        actions = np.stack(eval_dict_list[i]['action_list'])
        ctrl_interval = ctrl_interval_list[i]
        plt.plot(steps[::], actions[::, j], label=legend_list[i])
    plt.xlabel('Time Step')
    plt.ylabel(f"Action-{j+1}")
    plt.legend()
    plt.savefig(os.path.join(save_path, f"Action-{j+1}.png"))

# plot action error
# for j in range(action_dim):
#     plt.figure()
#     opt_actions = np.stack(eval_dict_list[0]['action_list'])
#     for i in range(1, len(opt_controllers)):
#         steps = eval_dict_list[i]['step_list']
#         action_errors = np.stack(eval_dict_list[i]['action_list']) - opt_actions
#         plt.plot(steps, action_errors[:, j], label=legend_list[i])
#     plt.xlabel('Time Step')
#     plt.ylabel(f"Action-{j+1} Error")
#     plt.legend()
#     plt.savefig(os.path.join(save_path, f"Action-{j+1} error.png"))

# plot state
for j in range(state_dim):
    plt.figure()
    for i in range(len(opt_controllers)):
        steps = eval_dict_list[i]['step_list']
        states = np.stack(eval_dict_list[i]['state_list'])
        plt.plot(steps, states[:, j], label=legend_list[i])
    plt.xlabel('Time Step')
    plt.ylabel(f"State-{j+1}")
    plt.legend()
    plt.savefig(os.path.join(save_path, f"State-{j+1}.png"))

# plot state error
# for j in range(state_dim):
#     plt.figure()
#     opt_states = np.stack(eval_dict_list[0]['state_list'])
#     for i in range(1, len(opt_controllers)):
#         steps = eval_dict_list[i]['step_list']
#         state_errors = np.stack(eval_dict_list[i]['state_list']) - opt_states
#         plt.plot(steps, state_errors[:, j], label=legend_list[i])
#     plt.xlabel('Time Step')
#     plt.ylabel(f"State-{j+1} Error")
#     plt.legend()
#     plt.savefig(os.path.join(save_path, f"State-{j+1} error.png"))

# plot reward
plt.figure()
for i in range(len(opt_controllers)):
    steps = eval_dict_list[i]['step_list']
    rewards = np.stack(eval_dict_list[i]['reward_list'])
    plt.plot(steps, rewards, label=legend_list[i])
plt.xlabel('Time Step')
plt.ylabel(f"Reward")
plt.legend()
plt.savefig(os.path.join(save_path, "Reward.png"))

for i in range(len(opt_controllers)):
    print(f"{legend_list[i]} time: {time_list[i]}")