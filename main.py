from time import time
import numpy as np
from utils import visualize
import controller as ctrl
import dynamics as dyn

# Simulation params
np.random.seed(10)
time_step = 0.5 # time between steps in seconds
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
obstacles = np.array([[-2,-2,0.5], [1,2,0.5]])

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
    cur_iter = 0 # iteration loop
    
    # Main loop
    while (cur_iter * time_step < sim_time): 
        t1 = time()

        # Get reference state
        cur_time = cur_iter*time_step
        cur_ref = dyn.lissajous(cur_iter, time_step)
        
        # Save current state and reference state for visualization
        ref_traj.append(cur_ref)
        car_states.append(cur_state)

        ################################################################
        # Generate control input
        # TODO: Replace this simple controller by your own controller
        control = ctrl.simple_controller(cur_state, cur_ref, v_min, v_max, w_min, w_max)
        print("[v,w]", control)
        ################################################################

        # Apply control input, Update current state
        next_state = dyn.car_next_state(time_step, cur_state, control, noise=True)
        cur_state = next_state
        
        # Loop time
        t2 = time()
        print(cur_iter)
        print(t2-t1)
        times.append(t2-t1)

        # accumulate error
        error_cur = cur_state - cur_ref
        error_cur[-1] = (error_cur[-1] + np.pi) % (2 * np.pi ) - np.pi # make theta error between -pi and pi
        error = error + np.linalg.norm(error_cur)
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
