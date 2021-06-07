import numpy as np


def lissajous(k, time_step):
    """This function returns the reference point at time step k

    Args:
        k (int): time index
        time_step (float): time resolution.
    Returns:
        list: [x_reference, y_reference, theta_ref]
    """
    xref_start = 0
    yref_start = 0
    A = 2
    B = 2
    a = 2*np.pi/50
    b = 3*a
    T = np.round(2*np.pi/(a*time_step))
    k = k % T
    delta = np.pi/2
    xref = xref_start + A*np.sin(a*k*time_step + delta)
    yref = yref_start + B*np.sin(b*k*time_step)
    v = [A*a*np.cos(a*k*time_step + delta), B*b*np.cos(b*k*time_step)]
    thetaref = np.arctan2(v[1], v[0]) # theta reference
    return [xref, yref, thetaref] 


def car_next_state(time_step, cur_state, control, noise = True):
    """implement car dynamics

    Args:
        time_step (int): current time index
        cur_state (np array): shape = (3,) -> [x,y,theta]
        control (np array): shape = (2,) -> [linear v, angular v]
        noise (bool, optional): Defaults to True.

    Returns:
        np array: shape = (3,) -> [x,y,theta]
    """
    theta = cur_state[2]
    rot_3d_z = np.array([[np.cos(theta), 0], [np.sin(theta), 0], [0, 1]])
    f = rot_3d_z @ control
    mu, sigma = 0, 0.04 # mean and standard deviation for (x,y)
    w_xy = np.random.normal(mu, sigma, 2)
    mu, sigma = 0, 0.004  # mean and standard deviation for theta
    w_theta = np.random.normal(mu, sigma, 1)
    w = np.concatenate((w_xy, w_theta))
    if noise:
        return cur_state + time_step*f.flatten() + w
    else:
        return cur_state + time_step * f.flatten()


def error_next_state(time_step, time_idx, current_error, control):
    """implement error dynamics

    Args:
        time_step (float): time step between difference samples
        time_idx (int): time index
        current_error (np array): shape = (3,) -> [x,y,theta]
        control (np array): shape = (2,) -> [linear v, angular v]

    Returns:
        np array: next_error, shape = (3,) -> [x,y,theta]
    """
    assert isinstance(time_step, float)
    assert isinstance(time_idx, int)

    # get reference trajectory
    gt_current = np.array(lissajous(time_idx, time_step)) # current ground truth
    gt_next = np.array(lissajous(time_idx+1, time_step))    # next ground truth

    # error dynamics
    theta = current_error[2] + gt_current[2]
    rot_3d_z = np.array([[np.cos(theta), 0], [np.sin(theta), 0], [0, 1]])
    f = rot_3d_z @ control # f.shape = (3,1)

    return current_error + time_step * f + (gt_current - gt_next)


def error_next_states(time_step, time_idx, X, U, res, x_range, y_range,theta_range):
    """Calculate next error states for all current error and controls.

    Args:
        time_step (float): tau, time resolution
        time_idx (int): time index
        X (np array): all error states, shape = (n_states, 3)
        U (np array): all controls, shape = (n_ctrl, 2)

    Returns:
        np array: next states, shape = (n_states, n_ctrl, 3)
    """
    assert isinstance(time_step, float)
    assert isinstance(time_idx, int)
    
    # get reference trajectory, groundtruth
    gt_current = np.array(lissajous(time_idx, time_step)) # gt_current.shape = (3,)
    gt_next = np.array(lissajous(time_idx+1, time_step))  # gt_next.shape = (3,)
    
    # rotation matrix
    theta = X[:,2] + gt_current[2]
    rot = np.zeros((X.shape[0],3,2))
    rot[:,0,0] = np.cos(theta)
    rot[:,1,0] = np.sin(theta)
    rot[:,2,1] = 1
    # print(f"rot{rot}")
    # calculate next error state 
    next = (rot * time_step) @ U.T # next.shape = (n_states, 3, n_ctrl)
    next = X[:,None,:] + next.transpose(0,2,1) + (gt_current - gt_next) # shape = (n_states, n_ctrl, 3)

    # print(f"raw : {next}")
    # snap to the grid
    next[:,:,:2] = np.around(next[:,:,:2] / res["xy"]) * res["xy"]
    next[:,:,2] = np.around(next[:,:,2] % (2*np.pi) / res["theta"]) * res["theta"]
 
    # clip to max, min
    np.clip(next[:,:,0], x_range[0], x_range[1], out=next[:,:,0])
    np.clip(next[:,:,1], y_range[0], y_range[1], out=next[:,:,1])
    np.clip(next[:,:,2], theta_range[0], theta_range[1], out=next[:,:,2])
    
    return np.around(next, decimals=3)