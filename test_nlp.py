# %%
import numpy as np
from casadi import *
import main
# %%
x = SX.sym('x'); y = SX.sym('y'); z = SX.sym('z')
f = x**2+100*z**2
nlp = {'x':vertcat(x,y,z), 'f':f, 'g':z+(1-x)**2-y}
opts = {'ipopt.print_level':0, 'print_time':0}
S = nlpsol('S', 'ipopt', nlp, opts)
print(S)
# %%
r = S(x0=[2.5,3.0,0.75],\
      lbg=0, 
      =0)
x_opt = r['x']
print('x_opt: ', x_opt)
# %%


# %%
test1 = main.lissajous(0)
# %%
def error_next_state(time_step, time_idx, current_error, control, lissajous):
    """implement car dynamics

    Args:
        time_step (float): time step between difference samples
        time_idx (int): time index
        current_error (np array): shape = (3,) -> [x,y,theta]
        control (np array): shape = (2,) -> [linear v, angular v]
        lissajous (function): function to preduct groundtruth trajectory

    Returns:
        np array: next_error, shape = (3,) -> [x,y,theta]
    """
    assert isinstance(time_step, float)
    assert isinstance(time_idx, int)

    # get reference trajectory
    gt_current = np.array(lissajous(time_idx)) # current ground truth
    gt_next = np.array(lissajous(time_idx+1))    # next ground truth

    # error dynamics
    theta = current_error[2] + gt_current[2]
    rot_3d_z = np.array([[np.cos(theta), 0], [np.sin(theta), 0], [0, 1]])
    f = rot_3d_z @ control # f.shape = (3,1)
    # print(f"f is {f}, f shape is {f.shape}")
    return current_error + time_step * f + (gt_current - gt_next)
# %%
time_step = 0.5
time_idx = 1
current_error = np.array([5,2,3])
control = [1,2]
lissajous = main.lissajous
next_error = error_next_state(time_step, time_idx, current_error, control, lissajous)
# %%

def get_total_cost(e_init, T, gamma, Q, q, R, U, time_step, time_init, lissajous):
    """Calculate total cost for CEC control

    Args:
        e_init (np array): shape (3,) error at time index = time_init
        T (int): CEC time horizon
        gamma (double): discount factor
        Q (np array): shape (2,2) weight for position error
        q (double): weight for angle error
        R (np array): shape (2,2) weight for control
        U (symbolic matrix): shape (2,T) control input
        time_step (double): time resolution
        time_init (int): time index
        lissajous (function): function to get ground truth

    Returns:
        symbolic: shape (1,1). total cost for horizon T
    """
    assert isinstance(T, int) and T > 0
    obj_func = 0
    current_error = e_init

    for t_idx in range(T):
        p_error = current_error[:2]
        theta_error = current_error[-1]
        u = U[:,t_idx]  # u.shape = (2,1)

        obj_func += gamma**t_idx * (p_error.T @ Q @ p_error + \
                                    q * (1 - np.cos(theta_error))**2 + \
                                    u.T @ R @ u)
        next_error = error_next_state(time_step, time_init+t_idx, current_error, u, lissajous)
        current_error = next_error

    # add terminal cost
    p_error = current_error[:2]
    theta_error = current_error[-1]
    obj_func += p_error.T @ Q @ p_error + q * (1 - np.cos(theta_error))**2
        
    return obj_func

def get_control_sym(T):
    U = []
    for t_idx in range(T):
        U.append(SX.sym(f'u_{t_idx}', 2))
    
    return U
# %%
U_test = get_control_sym(5)
# %%
T = 5
U = SX.sym('U', 2, T)
e_init = np.array([1,2,3])
gamma = 0.5
Q = np.eye(2)
q = 1
R = np.eye(2)
time_step = 0.5
time_init = 0
lissajous = main.lissajous

obj_func = get_total_cost(e_init, T, gamma, Q, q, R, U, time_step, time_init, lissajous)

# %%
# test NLP
u_upper = np.ones(U.shape) # u_upper.flatten(order="F")
u_lower = np.ones(U.shape) * np.array([0, -1]).reshape(-1,1)
# %%
