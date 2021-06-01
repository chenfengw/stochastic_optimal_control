# %%
import numpy as np
from casadi import *
import dynamics as dyn

# %%
x = SX.sym('x'); y = SX.sym('y'); z = SX.sym('z')
f = x**2+100*z**2
nlp = {'x':vertcat(x,y,z), 'f':f, 'g':z+(1-x)**2-y}
opts = {'ipopt.print_level':0, 'print_time':0}
S = nlpsol('S', 'ipopt', nlp, opts)
print(S)

r = S(x0=[2.5,3.0,0.75],\
      lbg=0, ubg=0)
x_opt = r['x']
print('x_opt: ', x_opt)

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
        next_error = dyn.error_next_state(time_step, time_init+t_idx, current_error, u, lissajous)
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
T = 5
U = SX.sym('U', 2, T)
e_init = np.array([1,2,3])
gamma = 0.5
Q = np.eye(2)
q = 1
R = np.eye(2)
time_step = 0.5
time_init = 0
lissajous = dyn.lissajous

obj_func = get_total_cost(e_init, T, gamma, Q, q, R, U, time_step, time_init, lissajous)

# %%
# test NLP
u_upper = np.ones(U.shape) # u_upper.flatten(order="F")
u_lower = np.ones(U.shape) * np.array([0, -1]).reshape(-1,1)
# %%
nlp = {'x':U[:], 'f':obj_func}
opts = {'ipopt.print_level':0, 'print_time':0}
S = nlpsol('S', 'ipopt', nlp, opts)
print(S)

# %%
r = S(x0 = np.random.rand(2*T), lbx=u_lower.flatten(order="F"), ubx=u_upper.flatten(order="F"))
x_opt = r['x']
print('x_opt: ', x_opt)
# %% test mutiple constrains
g = 9.8
N = 100

x = SX.sym('x',N)
v = SX.sym('v', N)
u = SX.sym('u', N-1)
#theta = SX('theta', N)
#thdot = SX('thetadot', N)

dt = 0.1
constraints = [x[0]-1, v[0]] # expressions that must be zero
for i in range(N-1):
    constraints += [x[i+1] - (x[i] + dt * v[i]) ]
    constraints += [v[i+1] - (v[i] - dt * x[i+1] + u[i] * dt)]

cost = sum([x[i]*x[i] for i in range(N)]) + sum([u[i]*u[i] for i in range(N-1)])

nlp = {'x':vertcat(x,v,u), 'f':cost, 'g':vertcat(*constraints)}


# %%
