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

env_id = "pyth_bypass"
algrithm = "DSAC2"
date_list = ["240427-170601"]
ite = "60000"
is_closed_loop = True

for date in date_list:
    runner = PolicyRunner(
        log_policy_dir_list=["../results/" + env_id + "/" + algrithm + "/" + date],
        trained_policy_iteration_list=[ite],
        is_init_info=True,
        # the initial position y_0 leads to a collision is not restrictly 0.0
        init_info={"init_state": [0.0, 0.0245466, 0.0, 0.0, 0, 0], "ref_time": 0.0,
                "ref_num": 13, "is_closed_loop": is_closed_loop, "surr_veh_dist": 10.0},
        save_render=True,
        legend_list=[algrithm + "/" + date + "/" + ite],
        constrained_env=True,
        is_tracking=True,
        dt=0.1
    )
    runner.run()
