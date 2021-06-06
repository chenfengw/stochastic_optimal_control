# %%
import numpy as np
import gym
import matplotlib.pyplot as plt; plt.ion()


class MDP(object):
  def __init__(self, P, nS, nA, gamma = 0.95, desc=None):
    """MDP object for markov decision process

    Args:
        P (dict): P[state][action] = [(prob, next_state 1, reward, done?),(prob, next_state 2, reward, done?)... ]
        nS (int): number of state
        nA (int): number of action
        gamma (float, optional): discont fector. Defaults to 0.95.
        desc ([type], optional): [description]. Defaults to None.
    """
    self.nS = nS                  # number of state
    self.nA = nA                  # number of actions
    self.gamma = gamma            # discount factor
    self.P = np.zeros((nS,nA,nS)) # transition probability: SxAxS' -> [0,1]
    self.L = np.zeros((nS,nA))    # stage cost: SxA -> R
    self.Y = np.full(nS,np.inf)   # terminal cost: S -> R       
    for s in range(nS):
      for a in range(nA):
        for (pr,ns,re,done) in P[s][a]:
          self.P[s,a,ns] += pr  # P[state, action, next state]
          self.L[s,a] -= re*pr
          if done: self.Y[ns] = 0  # set terminal_cost[terminal_state] = 0


def value_iteration(mdp, num_iter):
  """
  mdp: mdp object
  num_iter: number of iterations
  V, pi = value_iteration(mdp, num_iter)
  """
  # terminal and nontermina states
  term_sta = np.isfinite(mdp.Y)  # all terminal state
  ntrm_sta = ~term_sta           # all non terminal state
  
  # initialize the policy and value
  pi = np.zeros((num_iter+1,mdp.nS),dtype='int') # pi.shape = (number_iter+1, n_states)
  V = np.zeros((num_iter+1,mdp.nS))  # V.shape = (iter+1, n_states)
  V[:,term_sta] = mdp.Y[term_sta]    # set the value of the terminal states to terminal cost = 0
  
  # value iteration  
  for k in range(num_iter):
    Q = mdp.L[ntrm_sta,:] + mdp.gamma * np.sum(mdp.P[ntrm_sta,:,:] * V[k,None,None,:], axis=2) # Q.shape = (num_ntrm, n_action)
    pi[k+1,ntrm_sta] = np.argmin(Q, axis=1)    
    V[k+1,ntrm_sta] = np.min(Q,axis=1)

  return V, pi


def policy_iteration(mdp, num_iter):
  """
  mdp: mdp object
  num_iter: number of iterations
  Vpi, pi = policy_iteration(mdp, num_iter)
  """   
  # terminal and nontermina states  
  term_sta = np.isfinite(mdp.Y)       # all terminal state
  ntrm_sta = ~term_sta                # all non terminal state
  ntrm_I = np.eye(np.sum(ntrm_sta))   # eye(number of non terminal states)
  iall_sta = np.arange(mdp.nS)        # index of all state

  # initialize the policy and value
  pi = np.zeros((num_iter+1,mdp.nS),dtype='int')  # pi.shape = (n_iter+1, n_states)
  Vpi = np.zeros((num_iter+1,mdp.nS))             # V.shape = (n_iter+1, n_states)
  Vpi[:,term_sta] = mdp.Y[term_sta]               # set the value of the terminal states
  
  # policy iteration 
  for k in range(num_iter):
    
    # Policy Evaluation
    Ppi = mdp.P[iall_sta, pi[k]]   # Ppi.shape = (n_states, n_next_states)
    print(f"Ppi.shape {Ppi.shape}")
    A = ntrm_I - mdp.gamma * Ppi[ntrm_sta,:][:,ntrm_sta] # I - gama * P_nn, A.shape = (n_trm, n_trm)
    b = mdp.L[iall_sta, pi[k]][ntrm_sta] + Ppi[ntrm_sta,:][:,term_sta] @ mdp.Y[term_sta]
    Vpi[k,ntrm_sta] = np.linalg.solve(A, b)
    
    # Policy Improvement
    Qpi = mdp.L[ntrm_sta,:] + mdp.gamma * np.sum(mdp.P[ntrm_sta,:,:] * Vpi[k,None,None,:], axis=2)
    pi[k+1,ntrm_sta] = np.argmin(Qpi, axis=1) 

  
  # # Final Policy Evaluation
  # Ppi = mdp.P[iall_sta, pi[num_iter]]
  # A = ntrm_I - mdp.gamma * Ppi[ntrm_sta,:][:,ntrm_sta]
  # b = mdp.L[iall_sta, pi[k]][ntrm_sta] + Ppi[ntrm_sta,:][:,term_sta] @ mdp.Y[term_sta]
  # Vpi[num_iter,ntrm_sta] = np.linalg.solve(A, b)
  return Vpi, pi



def random_episode(env):
  print("    Let's look at a random episode...")
  env.reset()
  env.render()
  for t in range(100):
    a = env.action_space.sample()
    ob, re, done, prob = env.step(a)
    env.render()
    if done: break


def displayValuesText(V,pi):
  print("Iteration | max|V-Vprev| | # chg actions | V[0]")
  print("----------+--------------+---------------+---------")
  for k in range(V.shape[0]-1):
    max_diff = np.abs(V[k+1] - V[k]).max()
    nChgActions=(pi[k+1] != pi[k]).sum()
    print("%4i      | %6.5f      | %4s          | %6.5f"%(k+1, max_diff, nChgActions, V[k+1,0]))
  print("----------+--------------+---------------+---------\n")   


def displayValuesFig(V):
  plt.figure()
  plt.plot(V)
  plt.title("Values of different states")


def displayValueComparison(V1,V2):
  for s in range(5):
    plt.figure()
    plt.plot(np.array(V1)[:,s])
    plt.plot(np.array(V2)[:,s])
    plt.ylabel("value of state %i"%s)
    plt.xlabel("iteration")
    plt.legend(["value iteration", "policy iteration"], loc='best')

def displayActions(Vs,pis):
  for (V, pi) in zip(Vs[:10], pis[:10]):
    plt.figure(figsize=(3,3))
    plt.imshow(V.reshape(4,4), cmap='gray', interpolation='none', clim=(0,1))
    ax = plt.gca()
    ax.set_xticks(np.arange(4)-.5)
    ax.set_yticks(np.arange(4)-.5)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    Y, X = np.mgrid[0:4, 0:4]
    a2uv = {0: (-1, 0), 1:(0, -1), 2:(1,0), 3:(-1, 0)}
    Pi = pi.reshape(4,4)
    for y in range(4):
      for x in range(4):
        a = Pi[y, x]
        u, v = a2uv[a]
        plt.arrow(x, y,u*.3, -v*.3, color='m', head_width=0.1, head_length=0.1) 
        plt.text(x, y, str(env.desc[y,x].item().decode()),
                 color='g', size=12,  verticalalignment='center',
                 horizontalalignment='center', fontweight='bold')
    plt.grid(color='b', lw=2, ls='-')


# %%
env = gym.make("FrozenLake-v0")
#env.seed(0)

print("env.nS is the number of states.\n")
print("env.nA is the number of actions.\n")
print("env.P[state][action] is a list of tuples (probability, nextstate, reward, done).\n")
  
# Show a random episode
#random_episode(env)
  
# Create an MDP
mdp = MDP(env.P, env.nS, env.nA)

# Run Value Iteration
V1, pi1 = value_iteration(mdp, 130)
displayValuesText(V1,pi1)
displayValuesFig(V1)
#displayActions(V1,pi1)

# Run Policy Iteration
V2, pi2 = policy_iteration(mdp, 20)
displayValuesText(V2,pi2)
displayValuesFig(V2)
#displayActions(V2,pi2)

#displayValueComparison(V1,V2)


plt.show()

  




  

# %%

# %%
