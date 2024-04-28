## Description
The multimodal policy optimization (MUPO) algorithm is developed based on the General Optimal control Problem Solver (GOPS) framework by the Intelligent Driving Laboratory (iDLab).
The repository contains all environments, algorithms including baselines, trained network weights, and data used for plotting as described in the paper.

## Installation
GOPS requires:
1. Windows 7 or greater or Linux.
2. Python 3.6 or greater (GOPS V1.0 precompiled Simulink models use Python 3.6). We recommend using Python 3.8.
3. (Optional) Matlab/Simulink 2018a or greater.
4. The installation path must be in English.

You can install GOPS through the following steps:
```bash
# create conda environment
conda env create -f gops_environment.yml
conda activate gops
# install GOPS
pip install -e .
```

## Quick Start
This is an example of running MUPO on a vehicle tracking and obstacle avoidance environment called "bypass".
Train the policy by running:
```bash
python example_train/mupo_bypass.py
```
After training, test the policy by running:
```bash
python example_run/run_bypass.py
```
Then you can see the simulation results in the `figures` folder.
Remember to modify the model path in `run_bypass.py`, including the algorithm, date_list, iteration, and whether to use closed-loop dynamics (`is_closed_loop`).
When `is_closed_loop` is set to False, it uses an open-loop dynamics model, and the ego vehicle's state is updated based on the predefined state in `step_open_loop()` from the environment.
Figures 4c and 4d in the paper use the open-loop dynamics model, while other experiments use the closed-loop dynamics model.

Two examples of continuous policies colliding are provided here
```bash
python run_by_pass_collision.py
python run_encounter_collision.py
```

## Acknowledgment
We would like to thank all members in Intelligent Driving Laboratory (iDLab), School of Vehicle and Mobility, Tsinghua University for making excellent contributions and providing helpful advices for GOPS.