import numpy as np
import random

def epsilon_greedy_policy(Qtable, state, epsilon, env):
    random_int = random.uniform(0,1)
    if  random_int > epsilon:
        action = np.argmax(Qtable[state])
    else:
        action = env.action_space.sample()
    return action

def greedy_policy(Qtable, state):
    action = np.argmax(Qtable[state])
    return action