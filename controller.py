import numpy as np

def simple_controller(cur_state, ref_state, v_min, v_max, w_min, w_max):
    """Simple proportion controller

    Args:
        cur_state (np array): shape (3,)
        ref_state (list): [x_ref, y_ref, theta_ref] output from lissajous
        v_min (double): linear velocity min
        v_max (double): linear velocity max
        w_min (double): angular velocity min
        w_max (double): angular velocity max
    Returns:
        list: [linear_v, angular_v]
    """
    k_v = 0.55
    k_w = 1.0
    v = k_v*np.sqrt((cur_state[0] - ref_state[0])**2 + (cur_state[1] - ref_state[1])**2)
    v = np.clip(v, v_min, v_max)
    angle_diff = ref_state[2] - cur_state[2]
    angle_diff = (angle_diff + np.pi) % (2 * np.pi ) - np.pi
    w = k_w*angle_diff
    w = np.clip(w, w_min, w_max)
    return [v,w]

