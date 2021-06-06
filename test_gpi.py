# %%
import itertools
import numpy as np
import importlib
import dynamics
import value_iter
importlib.reload(dynamics)
importlib.reload(value_iter)
# %%
x_range = [-3,3]
y_range = [-3,3]
theta_range = [0, 2*np.pi]
v_range = [0,1]
w_range = [-1,1]
res = {"xy": 0.5, "theta": np.pi/5, "v": 0.1, "w":0.1}

# %% test get control space and state space 
state_space, state_dict, ctrl_space = value_iter.get_state_control_space(x_range, y_range, theta_range, v_range, w_range, res)

# %% test staget cost
Q = np.eye(2)
q = 1
R = np.eye(2)
# R = np.array([[0.93964841, 0.21466553],
#              [0.04171549, 0.93294541]])

X = np.array([[1,2,3], [4,5,6]])
U = np.array([[1,2],[3,4],[5,6],[7,8]])
state_cost = value_iter.calculate_stage_cost(X, U, Q, q, R)

# %% test get next_state, prototype
theta_test = np.arange(5)
rot = np.zeros((5,3,2))
rot[:,0,0] = np.cos(theta_test)
rot[:,1,0] = np.sin(theta_test)
rot[:,2,1] = 1

test = rot @ U.T
test = test.transpose(0,2,1) + np.arange(3)
add = np.vstack([np.arange(3)+1, 
                 np.arange(3)+2,
                 np.arange(3)+3,
                 np.arange(3)+4,
                 np.arange(3)+5])
test + add[:,None,:]

# %% 
time_step = 0.5
time_idx = 0
next_erros = dynamics.error_next_states(time_step, time_idx, X, U)
# %% test gaussian mean. this implements error dynamics. 
# essentially get next error
time_step = 0.5
t = 0
cur_error = np.array([1,2,3])
cur_control = np.array([[1,2],[3,4],[5,6],[4,5]])
means = value_iter.get_gaussian_mean(time_step, t, cur_error, cur_control, res)

# %% test adjacent states. these are states around the mean
adj_states, diff = value_iter.get_adjacent_states(means, res)

# test to convert adj states to tuple
adj_list = adj_states.tolist()
adj_list_tuple = [[tuple(item) for item in sublist] for sublist in adj_list]
adj_index = [[state_space[tuple(item)] for item in sublist] for sublist in adj_list]

#%% test gaussian transition probability
sigma = np.diag([0.04,0.04,0.004])**2
pdf = value_iter.gaussian_transition_prob(diff, sigma)

# %%
index = get_next_state_index(adj_states, state_space)
# %%
