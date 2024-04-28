#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University

#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com

#  Description: Create environments
#  Update Date: 2020-11-10, Yuhang Zhang: add create environments code

from gops.env.wrapper.wrapping_utils import wrapping_env
from gops.env.vector.sync_vector_env import SyncVectorEnv
from gops.env.vector.async_vector_env import AsyncVectorEnv
from gops.env.wrapper.gym2gymnasium import Gym2Gymnasium

def create_env(**kwargs):
    env_name = kwargs["env_id"]
    env_name_data = env_name + "_data"
    try:
        file = __import__(env_name_data)
    except NotImplementedError:
        raise NotImplementedError("This environment does not exist")

    env_name_camel = formatter(env_name)

    def env_fn():
        if hasattr(file, "env_creator"):
            env_class = getattr(file, "env_creator")
            env = env_class(**kwargs)
        elif hasattr(file, env_name_camel):
            env_class = getattr(file, env_name_camel)
            env = env_class(**kwargs)
        else:
            print("Env name: ", env_name_camel)
            raise NotImplementedError("This environment is not properly defined")

        # Wrapping the env
        max_episode_steps = kwargs.get("max_episode_steps", None)
        reward_scale = kwargs.get("reward_scale", None)
        reward_shift = kwargs.get("reward_shift", None)
        obs_scale = kwargs.get("obs_scale", None)
        obs_shift = kwargs.get("obs_shift", None)
        obs_noise_type = kwargs.get("obs_noise_type", None)
        obs_noise_data = kwargs.get("obs_noise_data", None)
        env = wrapping_env(
            env=env,
            max_episode_steps=max_episode_steps,
            reward_shift=reward_shift,
            reward_scale=reward_scale,
            obs_shift=obs_shift,
            obs_scale=obs_scale,
            obs_noise_type=obs_noise_type,
            obs_noise_data=obs_noise_data,
        )
        gym2gymnasium = kwargs.get("gym2gymnasium", False)
        if gym2gymnasium:
            env = Gym2Gymnasium(env)
        return env
    
    vector_env_num = kwargs.get("vector_env_num", None)
    vector_env_type = kwargs.get("vector_env_type", None)

    if vector_env_num is None:
        env = env_fn()
    else:
        env_fns = [env_fn] * vector_env_num
        if vector_env_type == "sync":
            env = SyncVectorEnv(env_fns)
        elif vector_env_type == "async":
            env = AsyncVectorEnv(env_fns)
        else:
            raise ValueError(f"Invalid vector_env_type {vector_env_type}!")
        
    print("Create environment successfully!")
    return env


def formatter(src: str, firstUpper: bool = True):
    arr = src.split("_")
    res = ""
    for i in arr:
        res = res + i[0].upper() + i[1:]

    if not firstUpper:
        res = res[0].lower() + res[1:]
    return res
