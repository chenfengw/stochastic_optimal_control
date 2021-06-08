# %%
from time import time
import numpy as np
from utils import visualize
import controller as ctrl
import dynamics as dyn

# Simulation params
np.random.seed(10)
time_step = 0.5  # time between steps in seconds
sim_time = 120    # simulation time, simulate data for 120 second

# Car params
x_init = 1.5
y_init = 0.0
theta_init = np.pi/2
v_max = 1
v_min = 0
w_max = 1
w_min = -1

# Obstacles in the environment
obstacles = np.array([[-2, -2, 0.5], [1, 2, 0.5]])

# CEC set up
cec_param = {}
cec_param["T"] = 5
cec_param["gamma"] = 0.9
cec_param["Q"] = np.eye(2) * 10
cec_param["q"] = 0.8
cec_param["R"] = np.eye(2)
cec_param["time_step"] = time_step
u_lower, u_upper, g_lower = ctrl.get_nlp_limits(
    v_min, 
    v_max, 
    w_min, 
    w_max, 
    obstacles, 
    cec_param["T"],
    r_inflation = 0.1
)

# %%
if __name__ == '__main__':
    # Params
    ref_traj = []
    error = 0.0
    car_states = []
    times = []

    # Start main loop
    main_loop = time()  # return time in sec
    # Initialize state
    cur_state = np.array([x_init, y_init, theta_init])
    cur_iter = 0  # iteration loop

    # Main loop
    while (cur_iter * time_step < sim_time):
        t1 = time()

        # Get reference state
        cur_time = cur_iter*time_step
        cur_ref = dyn.lissajous(cur_iter, time_step)
        cur_error = cur_state - cur_ref

        # Save current state and reference state for visualization
        ref_traj.append(cur_ref)
        car_states.append(cur_state)

        ################################################################
        # Generate control input
        # TODO: Replace this simple controller by your own controller
        # control = ctrl.simple_controller(cur_state, cur_ref, v_min, v_max, w_min, w_max)
        control = ctrl.controller_CEC(
            cur_error, 
            cur_iter, 
            cec_param, 
            u_lower, 
            u_upper, 
            g_lower, 
            obstacles
        )
        print(f"iter: {cur_iter}, [v,w]: {control}")
        ################################################################

        # Apply control input, Update current state
        next_state = dyn.car_next_state(time_step, cur_state, control, noise=True)
        cur_state = next_state

        # Loop time
        t2 = time()
        times.append(t2-t1)

        # accumulate error
        cur_error[-1] = (cur_error[-1] + np.pi) % (2 * np.pi) - np.pi  # make theta between [-pi, pi]
        error = error + np.linalg.norm(cur_error)
        cur_iter = cur_iter + 1

    main_loop_time = time()
    print('\n\n')
    print('Total time: ', main_loop_time - main_loop)
    print('Average iteration time: ', np.array(times).mean() * 1000, 'ms')
    print('Final error: ', error)

    # Visualization
    ref_traj = np.array(ref_traj)
    car_states = np.array(car_states)
    times = np.array(times)
    visualize(car_states, ref_traj, obstacles, times, time_step, save=True)
