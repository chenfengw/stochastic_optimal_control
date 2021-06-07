import numpy as np
import itertools
from dynamics import lissajous

def get_state_control_space(x_range, y_range, theta_range, v_range, w_range, res):
    """Get all control space and state space

    Args:
        x_range (list): range of x
        y_range (list): range of y
        theta_range (list): range of theta [theta_min, theta_max]
        v_range (list): range of linear velocity
        w_range (list): range of angular velocity
        res (dict): resolution for discretization

    Returns:
        tuple: (state_space, state2idx, ctrl_space)
        state_space -> np array. shape = (n_states, 3)
        state2idx -> dict, state to its index. state is tuple
        ctrl_space -> np array, control space, shape = (n_controls, 2)
    """
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
    idx2control = {idx: ctrl for idx, ctrl in enumerate(ctrl_space)}
    return np.array(state_space), state2idx, np.array(ctrl_space), idx2control


def calculate_stage_cost(X, U, Q, q, R):
    """Calculate stage cost given single error and all controls

    Args:
        X (np array): all error state. shape = (n_states, 3)
        U (np array): all controls. shape = (n_controls, 2)
        Q (np array): weight for xy error. shape = (2,2)
        q (float): weight for theta error
        R (np array): weight for controls. shape = (2,2)

    Returns:
        np array: stage cost. shape = (n_states, n_controls)
    """
    xy_cost = np.sum(X[:,:2].T * (Q @ X[:,:2].T), axis=0) # xy_cost.shape = (n_state,)
    theta_cost = q * (1 - np.cos(X[:,2]))**2       # theta_cost.shape = (n_state,)
    state_cost = xy_cost + theta_cost              # state_cost.shape = (n_state,)
    ctrl_cost = np.sum(U.T * (R @ U.T), axis=0)    # [u1.T @ R u1, u2.T @ R u2 ... ], shape = (n_controls,)
    
    # state_cost brocast column wise, ctrl_cost broadcast row wise
    return state_cost[:, np.newaxis] + ctrl_cost


def get_gaussian_mean(time_step, time_idx, current_error, control, res):
    """Get gaussian mean. Implement g(t,e,U,0). This is the same as error 
    dynamics.

    Args:
        time_step (float): time step between difference samples
        time_idx (int): time index
        current_error (np array): shape = (3,) -> [x,y,theta]
        control (np array): shape = (n_control, 2)
        res (dict): dictionary of resolution

    Returns:
        np array: means, shape = (n_control, 3). Next error state
    """
    assert isinstance(time_step, float)
    assert isinstance(time_idx, int)

    # get reference trajectory
    gt_current = np.array(lissajous(time_idx, time_step)) # gt_current.shape = (3,)
    gt_next = np.array(lissajous(time_idx+1, time_step))  # gt_next.shape = (3,)

    # error dynamics
    theta = current_error[2] + gt_current[2]
    rot_3d_z = np.array([[np.cos(theta), 0], [np.sin(theta), 0], [0, 1]])
    f = control @ rot_3d_z.T  # f.shape = (n_controls, 3)
    means = current_error + time_step * f + (gt_current - gt_next)
    means[:,-1] = means[:,-1] % (2*np.pi)

    # round to nearest grid
    means[:,0:2] = np.around(means[:,0:2]/res['xy']) * res['xy']
    means[:,2] = np.around(means[:,2]/res['theta']) * res['theta']
    
    return np.around(means, decimals=3)


def get_adjacent_states(means, res):
    """Calculate states adjacent to the mean. Using a 9 or 27 connected grids.

    Args:
        means (np array): shape = (n_controls, 3)
        res (dict): resolution

    Returns:
        tuple: (adj_states, diff)
        adj_states -> np array, shape = (n_controls, n_adj_states, 3). all adjacent states. 
        diff -> np array, shape = (n_adj, 3). Distance away from the mean. Same for all controls
    """
    xx, yy = np.meshgrid(np.array([-1,0,1]) * res["xy"], 
                         np.array([-1,0,1]) * res["xy"])
    diff = np.vstack((xx.ravel(), yy.ravel(), xx.ravel()* 0)).T  # diff.shape = (9,3)
    diff_rep = np.repeat(diff[:,np.newaxis,:], means.shape[0], axis=1) # diff_rep.shape = (9,n_control,3)
    adj_states = np.around(diff_rep + means, decimals=3)

    return adj_states.transpose(1,0,2), diff


def get_next_state_index(adj_states, state2idx):
    """get indexes of adjacent states

    Args:
        adj_states (np array): adjacent states to mean. shape = (n_control, n_adj_state, 3)
        state2idx (dict): dictionary that converts state to its index. 

    Returns:
        np array: index of all next states. shape = (n_ctrl, n_adj_state)
    """
    adj_index = [[state2idx[tuple(row)] for row in sub_array] for sub_array in adj_states]
    return np.array(adj_index)


def gaussian_transition_prob(X, sigma):
    """calculate gausssian transition probability. P(next_state | any control)
    The transition prob under any control is the same because we sample the next 
    state using the same pattern for all controls.

    Args:
        X (np array): mean centered states. shape = (n_next_state, 3). This is 
        basically distance away from the mean. You can use grids from get_adjacent_states
        sigma (np array): covariance matrix. shape = (3,3)

    Returns:
        np array: transition probability. shape = (n_next_state,)
        prob[i] = proability to go to next state i
    """
    assert X.shape[1] == sigma.shape[0] == sigma.shape[1]
    dist = np.sum(X.T * (np.linalg.inv(sigma) @ X.T), axis = 0) # calculate Mahalanobis distance
    n_dim = X.shape[1]
    coeff = 1 / np.sqrt((2*np.pi)**n_dim * np.linalg.det(sigma))
    prob = coeff * np.exp(-0.5 * dist)
    return prob / prob.sum(axis=0)