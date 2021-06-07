import importlib
import numpy as np
import casadi
import dynamics as dyn
import value_iter as vi

def simple_controller(cur_state, ref_state, v_min, v_max, w_min, w_max):
    """Simple proportion controller

    Args:
        cur_state (np array): shape (3,)
        ref_state (list): [x_ref, y_ref, theta_ref] output from lissajous
        v_min (float): linear velocity min
        v_max (float): linear velocity max
        w_min (float): angular velocity min
        w_max (float): angular velocity max
    Returns:
        list: [linear_v, angular_v]
    """
    k_v = 0.55
    k_w = 1.0
    v = k_v*np.sqrt((cur_state[0] - ref_state[0])**2 + (cur_state[1] - ref_state[1])**2)
    v = np.clip(v, v_min, v_max)
    angle_diff = ref_state[2] - cur_state[2]
    angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi # make angle_diff [-pi, pi]
    w = k_w*angle_diff
    w = np.clip(w, w_min, w_max)
    return [v,w]


def formulate_nlp(e_init, T, gamma, Q, q, R, U, time_step, time_init, obstacles):
    """Calculate total cost and constrain for CEC control

    Args:
        e_init (np array): shape (3,) error at time index = time_init
        T (int): CEC time horizon
        gamma (float): discount factor
        Q (np array): shape (2,2) weight for position error
        q (float): weight for angle error
        R (np array): shape (2,2) weight for control
        U (SX symbolic): shape (2,T) control input
        time_step (float): time resolution
        time_init (int): time index
        obstacles (np array): shape = (n_obsticales, 3). 
        Each row is [center_x, center_y, radius]
    
    Returns:
        tuple: objective_function (1x1 SX), contrains (list, len = 10)
    """
    assert isinstance(T, int) and T > 0
    assert isinstance(time_step, float)
    assert isinstance(time_init, int), "time init is time index"
    
    obj_func = 0
    current_error = e_init
    g_constrains = []

    for t_idx in range(T):
        p_error = current_error[:2]
        theta_error = current_error[-1]
        u = U[:,t_idx]  # u.shape = (2,1)

        obj_func += gamma**(t_idx) * (p_error.T @ Q @ p_error + \
                                      q * (1 - np.cos(theta_error))**2 + \
                                      u.T @ R @ u)
        next_error = dyn.error_next_state(time_step, time_init+t_idx, current_error, u)
        
        # calculate constrains
        next_state = next_error + dyn.lissajous(time_init+t_idx, time_step) # shape = (3,1), symbolic
        for i in range(obstacles.shape[0]):
            g_constrains += [(next_state[0] - obstacles[i][0])**2 + (next_state[1] - obstacles[i][1])**2]

        # set next error
        current_error = next_error # current_error.shape = (3,1)

    # add terminal cost
    p_error = current_error[:2]
    theta_error = current_error[-1]
    obj_func += p_error.T @ Q @ p_error + q * (1 - np.cos(theta_error))**2
        
    return obj_func, g_constrains


def get_nlp_limits(v_min, v_max, w_min, w_max, obstacles, T, r_inflation = 0.1):
    """Get lower, upper bound for variables needed for nlp.

    Args:
        v_min (float): linear velocity min
        v_max (float): linear velocity max
        w_min (float): angular velocity max
        w_max (float): angular v max
        obstacles (np array): shape = (n_obsticales, 3). Each row [centerX, centerY, radius]
        T (int): CEC time horizon
        r_inflation (float, optional): inflation for obsticles. Defaults to 0.1.

    Returns:
        tuple: u_lower_bound, u_upper_bound, g_lower_bound
    """
    # set limit for control u
    u_lower = np.ones((2,T)) * np.array([v_min, w_min]).reshape(-1,1)
    u_upper = np.ones((2,T)) * np.array([v_max, w_max]).reshape(-1,1)

    # set limit for constrains g
    g_lower = np.tile((obstacles[:,-1] + r_inflation)**2, T)
    
    return u_lower, u_upper, g_lower


def controller_CEC(error_cur, time_idx, param, u_lower, u_upper, g_lower, obstacles):
    """CEC controller. Output optimal control given current error state.

    Args:
        error_cur (np array): shape = (3,) error at current time
        time_idx (int): time index
        param (dict): parameter dictionary
        u_lower (np array): shape = (2,T) stored columnwise, lower bound for control
        u_upper (np array): shape = (2,T) stored columnwise, upper bound for control
        g_lower (np array): shape = (2*T,) lower bound for constrains 
        obstacles (np array): shape = (n_obsticales, 3). Each row [centerX, centerY, radius]

    Returns:
        list: optimal control [linear_velocity, angular_velocity]
    """
    U = casadi.SX.sym('U', 2, param["T"])
    obj_func, g_constrains = formulate_nlp(error_cur, 
                                           param["T"], 
                                           param["gamma"], 
                                           param["Q"], 
                                           param["q"], 
                                           param["R"], 
                                           U, 
                                           param["time_step"], 
                                           time_idx,
                                           obstacles)

    nlp = {'x': U[:], 'f':obj_func, 'g': casadi.vertcat(*g_constrains)}
    opts = {'ipopt.print_level':0, 'print_time':0}
    S = casadi.nlpsol('S', 'ipopt', nlp, opts)

    r = S(lbx=u_lower.flatten(order="F"), ubx=u_upper.flatten(order="F"), lbg=g_lower)
    x_opt = np.array(r['x'])
    return [x_opt[:2].item(0), x_opt[:2].item(1)]


def controller_VI(cur_error, time_idx, state_space, state_dict, ctrl_space, control_dict, time_step, res, ranges, cost_param, n_iter):

    # initialize the policy and value
    n_states = state_space.shape[0]
    n_controls = ctrl_space.shape[0]
    pi = np.zeros(n_states, dtype='int')
    V = np.zeros(n_states) 
  
    # calculate next states for t, next_errors.shape = (n_states, n_ctrl, 3)
    next_errors = dyn.error_next_states(time_step,
                                        time_idx, 
                                        state_space, 
                                        ctrl_space, 
                                        res,
                                        ranges["x_range"],
                                        ranges["y_range"],
                                        ranges["theta_range"])
    
    # calculate stage_costs, stage_costs.shape = (n_states, n_ctrl)
    stage_costs = vi.calculate_stage_cost(state_space, 
                                          ctrl_space, 
                                          cost_param["Q"], 
                                          cost_param["q"], 
                                          cost_param["R"])

    # calculate index of next_errors
    index = vi.get_next_state_index(next_errors, state_dict) # index.shape = (n_states, n_ctrl)
    index = index.ravel()
    
    # value iteration  
    for k in range(n_iter):
        Q = stage_costs + cost_param["gamma"] * V[index].reshape(n_states,-1) # Q.shape = (n_state, n_control)
        pi = np.argmin(Q, axis=1)    
        V = np.min(Q,axis=1)

    # convert current state to index
    cur_state = np.copy(cur_error)
    cur_state[:2] = np.around(cur_state[:2] / res["xy"]) * res["xy"]
    cur_state[2] = np.around(cur_state[2] % (2*np.pi) / res["theta"]) * res["theta"]
 
    # clip to max, min and get index of current state
    cur_state[0] = np.clip(cur_state[0], ranges["x_range"][0], ranges["x_range"][1])
    cur_state[1] = np.clip(cur_state[1], ranges["y_range"][0], ranges["y_range"][1])
    cur_state[2] = np.clip(cur_state[2], ranges["theta_range"][0], ranges["theta_range"][1])
    cur_state = np.around(cur_state, decimals=3)
    state_idx = state_dict[tuple(cur_state)]

    # get optimal control index
    ctrl_idx = pi[state_idx]
    return list(control_dict[ctrl_idx])
