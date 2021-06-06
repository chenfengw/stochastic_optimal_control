# %%
import itertools
import numpy as np
x_range = [-3,3]
y_range = [-3,3]
theta_range = [0, 2*np.pi]
v_range = [0,1]
w_range = [-1,1]
res = {"xy": 0.2, "theta": 5*np.pi/180, "v": 0.1, "w":0.1}

# %%
def get_state_control_space(x_range, y_range, theta_range, v_range, w_range, res):
    all_x = np.around(np.arange(x_range[0], x_range[1]+res["xy"], res["xy"]), decimals=3)
    all_y = np.around(np.arange(y_range[0],y_range[1]+res["xy"], res["xy"]), decimals=3)
    all_theta = np.around(np.arange(theta_range[0], theta_range[1]+res["theta"], res["theta"]), decimals=3)
    all_v = np.around(np.arange(v_range[0], v_range[1]+res["v"], res["v"]), decimals=3)
    all_w = np.around(np.arange(w_range[0], w_range[1]+res["w"], res["w"]), decimals=3)

    # get control space
    state_space = list(itertools.product(all_x, all_y, all_theta))
    ctrl_space = list(itertools.product(all_v, all_w))

    # get dictionary for index
    state2idx = {state: idx for idx, state in enumerate(state_space)}
    ctrl2idx = {ctrl: idx for idx, ctrl in enumerate(ctrl_space)}
    return state_space, ctrl_space

# %%
state_space, ctrl_space = get_state_control_space(x_range, y_range, theta_range, v_range, w_range, res)

# %%
def calculate_stage_cost(error, U, Q, q, R):
    """Calculate stage cost given single error and all controls

    Args:
        error (np array): single error state. shape = (3,)
        U (np array): all controls. shape = (n_controls, 2)
        Q (np array): weight for xy error. shape = (2,2)
        q (float): weight for theta error
        R (np array): weight for controls. shape = (2,2)

    Returns:
        np array: stage cost. shape = (n_controls,)
    """
    error1 = error[:2].T @ Q @ error[:2] + q * (1 - np.cos(error[2]))**2
    error2 = np.sum(U * U @ R, axis=1) # [u1.T @ R u1, u2.T @ R u2 ... ]
    return error1 + error2
    
# %%
Q = np.eye(2)
q = 1
R = np.eye(2)
# R = np.array([[0.09782802, 0.30811832],
#        [0.15060282, 0.3939782 ]])

error = np.array([1,2,3])
U = np.array([[1,2],[3,4],[5,6]])
error1, error2 = calculate_stage_cost(error, U, Q, q, R)

# %%
from dynamics import get_gaussian_mean
time_step = 0.5
t = 0
cur_error = np.array([1,2,3])
cur_control = np.array([[1,2],[3,4],[5,6],[4,5]])
means = get_gaussian_mean(time_step, t, cur_error, cur_control, res)
# %%
nx, ny = (3, 3)
x = np.linspace(0, 1, nx)
y = np.linspace(0, 1, ny)
xv, yv = np.meshgrid(x, y)
# %%
xx, yy = np.meshgrid([-1,0,1], [-1,0,1])
# %%
test = np.vstack((xx.ravel(), yy.ravel(), xx.ravel()* 0)).T
# %%
test2 = np.repeat(test[np.newaxis,:,:], 2, axis=0)
# %%

def get_adjacent_states(means, res):
    """Calculate states adjacent to the mean.

    Args:
        means (np array): shape = (n_controls, 3)
        res (dict): discrimination resolution

    Returns:
        np array: all adjacent states shape = (n_controls, n_adjacent_state, 3)
    """
    xx, yy = np.meshgrid(np.array([-1,0,1]) * res["xy"], 
                         np.array([-1,0,1]) * res["xy"])
    grids = np.vstack((xx.ravel(), yy.ravel(), xx.ravel()* 0)).T  # grids.shape = (9,3)
    grids_rep = np.repeat(grids[:,np.newaxis,:], means.shape[0], axis=1) # grids_rep.shape = (9,n_control,3)
    adj_states = grids_rep + means
    return adj_states.transpose(1,0,2)
# %%
adj_states = get_adjacent_states(means, res)
# %%
xx, yy = np.meshgrid(np.array([-1,0,1]) * res["xy"], 
                         np.array([-1,0,1]) * res["xy"])
grids = np.vstack((xx.ravel(), yy.ravel(), xx.ravel()* 0)).T  # grids.shape = (9,3)
# %%
grids_rep = np.repeat(grids[:,np.newaxis,:],2, axis=1)
# %%
