import numpy as np
import gym
import random
import imageio

from utils.initialize import initialize_q_table
from train import train
from evaluate import evaluate_agent

env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False)

state_space = env.observation_space.n
print("There are ", state_space, " possible states")
action_space = env.action_space.n
print("There are ", action_space, " possible actions")

Qtable_frozenlake = initialize_q_table(state_space, action_space)

n_training_episodes = 10000

max_steps = 99
n_eval_episodes = 100
eval_seed = []

max_epsilon = 1.0
min_epsilon = 0.05
decay_rate = 0.0005

Qtable_frozenlake = train(n_training_episodes, min_epsilon, max_epsilon, decay_rate, env, max_steps, Qtable_frozenlake)

print(Qtable_frozenlake)

mean_reward, std_reward = evaluate_agent(env, max_steps, n_eval_episodes, Qtable_frozenlake, eval_seed)
print(f"Mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")

