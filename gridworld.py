# grid_world.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to Clemson University and the authors.
# 
# Authors: Pei Xu (peix@g.clemson.edu) and Ioannis Karamouzas (ioannis@g.clemson.edu)

"""
In this assignment, you will implement three classic algorithm for 
solving Markov Decision Processes either offline or online. 
These algorithms include: value_iteration, policy_iteration and q_learning.
You will test your implementation on three grid world environments. 
You will also have the opportunity to use Q-learning to control a simulated robot 
in crawler.py

The package `matplotlib` is needed for the program to run.

The Grid World environment has discrete state and action spaces
and allows for both model-based and model-free access.

It has the following properties:
    env.observation_space.n     # the number of states
    env.action_space.n          # the number of actions
    env.trans_model             # the transition/dynamics model

In value_iteration and policy_iteration, you can access the transition model 
at a given state s and action by calling
    t = env.trans_model[s][a]
where s is an integer in the range [0, env.observation_space.n),
      a is an integer in the range [0, env.action_space.n), and
      t is a list of four-element tuples in the form of
        (p, s_, r, terminal)
where s_ is a new state reachable from the state s by taking the action a,
      p is the probability to reach s_ from s by a, i.e. p(s_|s, a),
      r is the reward of reaching s_ from s by a, and
      terminal is a boolean flag to indicate if s_ is a terminal state.

In q_learning, once a terminal state is reached the environment should be (re)initialized by
    s = env.reset()
where s is the initial state.
An experience (sample) can be collected from s by taking an action a as follows:
    s_, r, terminal, info = env.step(a)
where s_ is the resulted state by taking the action a,
      r is the reward achieved by taking the action a,
      terminal is a boolean flag to indicate if s_ is a terminal state, and
      info is just used to keep compatible with openAI gym library.


A Logger instance is provided for each function, through which you can
visualize the process of the algorithm.
You can visualize the value, v, and policy, pi, for the i-th iteration by
    logger.log(i, v, pi)
You can also only update the visualization of the v values by
    logger.log(i, v)
"""


# use random library if needed
import random


def value_iteration(env, gamma, max_iterations, logger):
    """
    Implement value iteration to return a deterministic policy for all states.

    Parameters
    ----------
    env: GridWorld
        the environment
    gamma: float
        the reward discount factor
    max_iterations: integer
        the maximum number of value iterations that should be performed;
        the algorithm should terminate when max_iterations is exceeded.
        Hint: The value iteration may converge before reaching max_iterations.  
        In this case, you may want to exit the algorithm earlier. A way to check 
        if value iteration has already converged is to check whether 
        the max over (or sum of) L1 or L2 norms between the values before and
        after an iteration is small enough. For the Grid World environment, 1e-4
        is an acceptable tolerance.
    logger: app.grid_world.App.Logger
        a logger instance to perform test and record the iteration process
    
    Returns
    -------
    pi: list or dict
        pi[s] should give a valid action,
        i.e. an integer in [0, env.action_space.n),
        as the optimal policy found by the algorithm for the state s.
    """
    NUM_STATES = env.observation_space.n
    NUM_ACTIONS = env.action_space.n
    TRANSITION_MODEL = env.trans_model

    v = [0] * NUM_STATES
    pi = [0] * NUM_STATES
    # Visualize the value and policy 
    logger.log(0, v, pi)
    # At each iteration, you may need to keep track of pi to perform logging
   
### Please finish the code below ##############################################
###############################################################################
    def lookahead(state, v):
        expected_Action = [0] * NUM_ACTIONS
        for act in range(NUM_ACTIONS):
            for prob, next_state, reward, terminal in TRANSITION_MODEL[state][act]:
                if terminal:
                    g = 0
                else:
                    g =1
                expected_Action[act] += prob * (reward + g*gamma * v[next_state])
                
        return expected_Action
    
    def argmax(a):
        return max(range(len(a)), key=lambda x: a[x])
    
    for k in range(0, max_iterations):
        threshold = 0      #for Stopping condition
        for curr_state in range(NUM_STATES):
            possible_act = lookahead(curr_state, v)   #look ahead to find the best action
            best_action = max(possible_act)
            threshold = max(threshold, abs(best_action - v[curr_state]))
            v[curr_state] = best_action

        #Stopping condition given as 1e-4
        if threshold < 0.000001: 
            break

        #logger.log(k+1,v,pi)

    #To calculate the optimal policy using only the valid actions (best action possible)
        for s in range(NUM_STATES):
            action = lookahead(s, v)
            best = argmax(action)
            pi[s] = best


        logger.log(k,v,pi)
    
###############################################################################
    return pi


def policy_iteration(env, gamma, max_iterations, logger):
    """
    Implement policy iteration to return a deterministic policy for all states.

    Parameters
    ----------
    env: GridWorld
        the environment
    gamma: float
        the reward discount factor
    max_iterations: integer
        the maximum number of policy iterations that should be performed;
        the algorithm should terminate when max_iterations is exceeded.
        Hint 1: Policy iteration may converge before reaching max_iterations. 
        In this case, you should exit the algorithm. A simple way to check 
        if the algorithm has already converged is by simply checking whether
        the policy at each state hasn't changed from the previous iteration.
        Hint 2: The value iteration during policy evaluation usually converges 
        very fast and policy evaluation should end upon convergence. A way to check 
        if policy evaluation has converged is to check whether the max over (or sum of) 
        L1 or L2 norm between the values before and after an iteration is small enough. 
        For the Grid World environment, 1e-4 is an acceptable tolerance.
    logger: app.grid_world.App.Logger
        a logger instance to record and visualize the iteration process.
        During policy evaluation, the V-values will be updated without changing the current policy; 
        here you can update the visualization of value by simply calling logger.log(i, v).
    
    Returns
    -------
    pi: list or dict
        pi[s] should give a valid action,
        i.e. an integer in [0, env.action_space.n),
        as the optimal policy found by the algorithm for the state s.
    """
    NUM_STATES = env.observation_space.n
    NUM_ACTIONS = env.action_space.n
    TRANSITION_MODEL = env.trans_model
    
    v = [0.0] * NUM_STATES
    pi = [random.randint(0, NUM_ACTIONS-1)] * NUM_STATES
    # Visualize the initial value and policy
    logger.log(0, v, pi)

### Please finish the code below ##############################################
###############################################################################
    def p_eval(curr_iter, pi, env, gamma, max_iterations):
        for i in range(0, max_iterations):
            threshold = 0
            v_old = v.copy()
            for curr_state in range(NUM_STATES):
                expected_val = 0
                #To look for next possible actions
                    #To look for next possible states for each action
                for prob, next_state, reward, terminal in TRANSITION_MODEL[curr_state][pi[curr_state]]:
                    #To calculate the expected value
                    if terminal:
                        g = 0
                    else:
                        g =1
                    expected_val += prob * (reward + g*gamma * v_old[next_state])
                        
                threshold = max(threshold, abs(v_old[curr_state] - expected_val))
                v[curr_state] = expected_val
            logger.log(curr_iter, v)
            #To calculate the stopping condition:
            if threshold < 0.000001:
                break
        return v

    #Lookahead functions calculates the value for all actions in a given state. It is copied from value iteration.
    def lookahead(state, v):
        expected_Action = [0] * NUM_ACTIONS
        for act in range(NUM_ACTIONS):
            for prob, next_state, reward, terminal in TRANSITION_MODEL[state][act]:
                if terminal:
                    g = 0
                else:
                    g =1
                expected_Action[act] += prob * (reward + g*gamma * v[next_state])
                
        return expected_Action
    
    # This function returns the max value, i.e., to return the best possible action.
    def argmax(a):
        return max(range(len(a)), key=lambda x: a[x])
    
    #pi is assigned with a random value, This is to assign random policy

    for i in range(0, max_iterations):
        # To Evaluate current policy:
        v = p_eval(i, pi, env, gamma, max_iterations)

        #Flag to indicate any changes to the policy. Set to False in case of changes.
        stable_flag =  True

        for curr_state in range(NUM_STATES):
            curr_action = pi[curr_state]

            #To find the best possible action
            action_val = lookahead(curr_state, v)
            best_action = argmax(action_val)

            #To update the policy
            if curr_action != best_action:
                stable_flag = False

            pi[curr_state] = best_action

            
        logger.log(i,v,pi)
        
        if stable_flag:
            return pi

###############################################################################
    
import numpy as np
def q_learning(env, gamma, max_iterations, logger):
    """
    Implement Q-learning to return a deterministic policy for all states.

    Parameters
    ----------
    env: GridWorld
        the environment
    gamma: float
        the discount factor
    max_iterations: integer
        the maximum number of iterations (training episodes) that should be performed;
        the algorithm should terminate when max_iterations is exceeded.
    logger: app.grid_world.App.Logger
        a logger instance to perform test and record the iteration process.
    
    Returns
    -------
    pi: list or dict
        pi[s] should give a valid action,
        i.e. an integer in [0, env.action_space.n),
        as the optimal policy found by the algorithm for the state s.
    """
    NUM_STATES = env.observation_space.n
    NUM_ACTIONS = env.action_space.n
    
    v = [0] * NUM_STATES
    pi = [0] * NUM_STATES
    # Visualize the initial value and policy
    logger.log(0, v, pi)

    #########################
    # Adjust superparameters as you see fit
    #
    # parameter for the epsilon-greedy method to trade off exploration and exploitation
    eps_initial = 1
    # learning rate for updating q values based on sample estimates
    alpha = 0.1
    #########################

    

### Please finish the code below ##############################################
###############################################################################

#initialition of all state actions for zero
    Q=np.zeros((NUM_STATES,NUM_ACTIONS))
    def epsilon_greedy(Q, eps, NUM_ACTIONS, s):
        if np.random.rand() < eps:
            action = np.random.randint(0, NUM_ACTIONS)
        else:
            action = np.argmax(Q[s, :])
        
        return action
#q-learning iteration upto max_iteration
    s=env.reset()
    for i in range(max_iterations):
        eps_final=0.1
        #decrease eps as with increase in iteration
        iteration =min(0.9*max_iterations, i)
        eps = (eps_final * iteration + eps_initial * (0.9*max_iterations - iteration))/(0.9*max_iterations)
        #choose action
        a = epsilon_greedy(Q, eps, NUM_ACTIONS, s)
        s_, r, terminal, info = env.step(a)
        #set reward policy and v-values
        if terminal:
            estimate=r
            Q[s, a] =((1-alpha)*Q[s,a]) + (alpha * estimate)
            pi[s] =np.argmax(Q[s,:])
            logger.log(i, v, pi)
            s = env.reset()
        else:
            a_ = np.argmax(Q[s_, :])
            estimate=r+gamma*Q[s_,a_]
            Q[s, a] = ((1-alpha)*Q[s,a]) + (alpha * estimate) 
            v[s] = max(Q[s])
            pi[s] =np.argmax(Q[s,:])
            s, a = s_, a_

###############################################################################
    return pi


if __name__ == "__main__":
    from app.grid_world import App
    import tkinter as tk

    algs = {
        "Value Iteration": value_iteration,
        "Policy Iteration": policy_iteration,
        "Q Learning": q_learning
   }
    worlds = {
        # o for obstacle
        # s for start cell
        "world1": App.DEFAULT_WORLD,
        "world2": lambda : [
            [10, "s", "s", "s", 1],
            [-10, -10, -10, -10, -10],
        ],
        "world3": lambda : [
            ["_", "_", "_", "_", "_"],
            ["_", "o", "_", "_", "_"],
            ["_", "o",   1, "_",  10],
            ["s", "_", "_", "_", "_"],
            [-10, -10, -10, -10, -10]
        ]
    }

    root = tk.Tk()
    App(algs, worlds, root)
    tk.mainloop()
