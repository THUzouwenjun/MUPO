import numpy as np

# # Add do_mpc to path. This is not necessary if it was installed via pip.
# import sys
# sys.path.append('../../')

# Import do_mpc package:
import do_mpc
import casadi as ca
from casadi import *

model_type = 'discrete' # either 'discrete' or 'continuous'
model = do_mpc.model.Model(model_type)

x_1 = model.set_variable(var_type='_x', var_name='x_1', shape=(1,1))
x_2 = model.set_variable(var_type='_x', var_name='x_2', shape=(1,1))
x_3 = model.set_variable(var_type='_x', var_name='x_3', shape=(1,1))
x_4 = model.set_variable(var_type='_x', var_name='x_4', shape=(1,1))
# Two states for the desired (set) motor position:
phi_m_1_set = model.set_variable(var_type='_u', var_name='u_1')
phi_m_2_set = model.set_variable(var_type='_u', var_name='u_2')

# As shown in the table above, we can use Long names or short names for the variable type.
Theta_1 = model.set_variable('parameter', 'Theta_1') 
Theta_2 = model.set_variable('parameter', 'Theta_2')
Theta_3 = model.set_variable('parameter', 'Theta_3')

c = np.array([2.697,  2.66,  3.05, 2.86])*1e-3
d = np.array([6.78,  8.01,  8.82])*1e-5

# model.set_rhs('phi_1', dphi[0])
# model.set_rhs('phi_2', dphi[1])
# model.set_rhs('phi_3', dphi[2])

A=[
    [0, 1, 0, 0],
    [0, 1, 0, 0],
    [0.1, -0.2, 0, 0.5],
    [-0.2, 0.1, 0.1, 0]
]
B=[
    [0, 0],
    [-2, -1],
    [0.0, 0],
    [1, 1.5]
]
dt = 0.1
A0 = np.array(A).astype('float64')
A = np.linalg.pinv(np.eye(A0.shape[0]) - A0 * dt)
B0 = np.array(B).astype('float64')
B = A @ B0 * dt

A = ca.DM(A)
B = ca.DM(B)

# A = ca.vertcat(
#     ca.horzcat(0, 1, 0, 0),
#     ca.horzcat(0, 1, 0, 0),
#     ca.horzcat(0.1, -0.2, 0, 0.5),
#     ca.horzcat(-0.2, 0.1, 0.1, 0)
# )

# B = ca.vertcat(
#     ca.horzcat(0, 0),
#     ca.horzcat(-2, -1),
#     ca.horzcat(0, 0),
#     ca.horzcat(1, 1.5)
# )

Q = ca.diagcat(1, 2, 2, 1)

R = ca.diagcat(1.0, 1.0)



delta_x = vertcat(
    
)

model.set_rhs('x_1', A[0] )
model.set_rhs('x_2', 1/tau*(phi_m_1_set - phi_1_m))
model.set_rhs('x_3', 1/tau*(phi_m_2_set - phi_2_m))

model.setup()

mpc = do_mpc.controller.MPC(model)

setup_mpc = {
    'n_horizon': 20,
    't_step': 0.1,
    'n_robust': 1,
    'store_full_solution': True,
}
mpc.set_param(**setup_mpc)

mterm = phi_1**2 + phi_2**2 + phi_3**2
lterm = phi_1**2 + phi_2**2 + phi_3**2

mpc.set_objective(mterm=mterm, lterm=lterm)

mpc.set_rterm(
    phi_m_1_set=1e-2,
    phi_m_2_set=1e-2
)

# Lower bounds on states:
mpc.bounds['lower','_x', 'phi_1'] = -2*np.pi
mpc.bounds['lower','_x', 'phi_2'] = -2*np.pi
mpc.bounds['lower','_x', 'phi_3'] = -2*np.pi
# Upper bounds on states
mpc.bounds['upper','_x', 'phi_1'] = 2*np.pi
mpc.bounds['upper','_x', 'phi_2'] = 2*np.pi
mpc.bounds['upper','_x', 'phi_3'] = 2*np.pi

# Lower bounds on inputs:
mpc.bounds['lower','_u', 'phi_m_1_set'] = -2*np.pi
mpc.bounds['lower','_u', 'phi_m_2_set'] = -2*np.pi
# Lower bounds on inputs:
mpc.bounds['upper','_u', 'phi_m_1_set'] = 2*np.pi
mpc.bounds['upper','_u', 'phi_m_2_set'] = 2*np.pi

mpc.setup()