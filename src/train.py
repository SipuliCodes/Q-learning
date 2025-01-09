import numpy as np
from tqdm.notebook import trange

from policy import epsilon_greedy_policy

n_training_episodes = 10000
learning_rate = 0.7

n_eval_episodes = 100

env_id = "FrozenLake-v1"
max_steps = 99
gamma = 0.95
eval_seed = []

def train(n_training_episodes, min_epsilon, max_epsilon, decay_rate, env, max_steps, Qtable):
    for episode in trange(n_training_episodes):
        epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode)

        state = env.reset()
        step = 0
        done = False

        for step in range(max_steps):

            action = epsilon_greedy_policy(Qtable, state, epsilon, env)

            new_state, reward, done, info = env.step(action)

            Qtable[state][action] = Qtable[state][action] + learning_rate * (reward + gamma * np.max(Qtable[new_state]) - Qtable[state][action])

            if done:
                break

            state = new_state

    return Qtable