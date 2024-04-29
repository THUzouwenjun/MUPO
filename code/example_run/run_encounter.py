#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: run a closed-loop system
#  Update: 2022-12-05, Congsheng Zhang: create file
#  Update: 2024-04-28, Wenjun Zou: update file

from gops.sys_simulator.sys_run import PolicyRunner

env_id = "pyth_encounter"
algrithm = "MUPO"
date_list = ["240428-152509"]
ite = "60000"
is_closed_loop = True

for date in date_list:
    runner = PolicyRunner(
        log_policy_dir_list=["../results/" + env_id + "/" + algrithm + "/" + date],
        trained_policy_iteration_list=[ite],
        is_init_info=True,
        # Here's a breakdown of the state variables:
        # x - position along the x-axis (in meters),
        # y - position along the y-axis (in meters),
        # phi - yaw angle (in radians),
        # vx - longitudinal velocity (in meters per second),
        # vy - lateral velocity (in meters per second),
        # r - yaw rate (in radians per second)
        # "ref_num": 13 means the straight path. We only trained on straight paths.
        init_info={"init_state": [0.10-0.01, 0.0, 0.0, 0.0, 0, 0], "ref_time": 0.0,
                "ref_num": 13, "is_closed_loop": is_closed_loop},
        save_render=True,
        legend_list=[algrithm + "/" + date + "/" + ite],
        constrained_env=True,
        is_tracking=True,
        dt=0.1
    )

    runner.run()
