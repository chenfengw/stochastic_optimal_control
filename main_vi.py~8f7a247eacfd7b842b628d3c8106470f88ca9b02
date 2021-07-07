# %%
from time import time
import numpy as np
from utils import visualize
import controller as ctrl
import dynamics as dyn
import value_iter as vi
# Simulation params
np.random.seed(10)
time_step = 0.5  # time between steps in seconds
# sim_time = 120    # simulation time, simulate data for 120 second
sim_time = 100

# Car initialization
x_init = 1.5
y_init = 0.0
theta_init = np.pi/2

# Obstacles in the environment
obstacles = np.array([[-2, -2, 0.5], [1, 2, 0.5]])

# discretization set up
ranges = {}
ranges["x_range"] = [-1.5,1.5]
ranges["y_range"] = [-1.5,1.5]
ranges["theta_range"] = [0, 2*np.pi]
ranges["v_range"] = [0,1]
ranges["w_range"] = [-1,1]
res = {"xy": 0.5, "theta": np.pi/30, "v": 0.1, "w":0.1}

# get all state and control space
state_space, state_dict, ctrl_space, control_dict = vi.get_state_control_space(
    ranges["x_range"], 
    ranges["y_range"], 
    ranges["theta_range"], 
    ranges["v_range"], 
    ranges["w_range"], 
    res)

# set cost parmeters
cost_param = {}
cost_param["gamma"] = 0.9
cost_param["Q"] = np.eye(2) * 10
cost_param["q"] = 0.8
cost_param["R"] = np.eye(2)

# calculate stage_costs, stage_costs.shape = (n_states, n_ctrl)
stage_costs = vi.calculate_stage_cost(state_space, 
                                      ctrl_space, 
                                      cost_param["Q"], 
                                      cost_param["q"], 
                                      cost_param["R"])

# value iteration n loop
n_iter = 50

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
        control = ctrl.controller_VI(cur_error, cur_iter, state_space, state_dict, ctrl_space, control_dict, stage_costs, time_step, res, ranges, cost_param, n_iter)
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
