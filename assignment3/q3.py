import gym
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
from utils import *

import sys
import logging as log
import time
from actor_critic_agent_ import ActorCriticAgent, PolicyNetwork
from pnn import PNNAgent
from utils import *

np.random.seed(1)
N_ITERS = 10
tf.disable_v2_behavior()


def train_transfer(source_env, target_env):
    agent = ActorCriticAgent(env=gym.make(source_env), render=True)
    agent.set_env(source_env)

    start = time.time()
    solved, curr_ep, last_score, v_loss, p_loss = agent.train()
    end = time.time()

    save_losses(curr_ep, v_loss, p_loss, f'{source_env}_source', label=f'{source_env} - Source', time=end-start)

    if solved:
        agent.set_env(target_env)
        agent.init_output_weights()
        start = time.time()
        solved, curr_ep, last_score, v_loss, p_loss = agent.train()
        end = time.time()
        save_losses(curr_ep, v_loss, p_loss, f'{target_env}_target', label=f'{target_env} - Target', time=end-start)


def run_pnn(target_env_name, sources, target):
    agent = PNNAgent(env=gym.make(target_env_name), source_networks=sources, target_network=target)

    solved, curr_ep, last_score, v_loss, p_loss, duration = agent.train()

    save_losses(curr_ep, v_loss, p_loss, f'{target_env_name}_target_pnn', label=f'{target_env_name} - Target - PNN', time=duration)


if __name__ == '__main__':
    agent = ActorCriticAgent(env=gym.make(ENVS['cp']), render=True)
    solved, _, _, _, _, _ = agent.train()
    if solved:
        source = agent.policy
        target = PolicyNetwork()
        run_pnn(ENVS['acro'], source, target)

