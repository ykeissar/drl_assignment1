import gym
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
from utils import *

import sys
import logging as log
import time
from actor_critic_agent_ import ActorCriticAgent


np.random.seed(1)
N_ITERS = 10
tf.disable_v2_behavior()


def only_one(env_name):
    agent = ActorCriticAgent(env=gym.make(env_name), render=True)
    agent.set_env(env_name)
    start = time.time()
    solved, curr_ep, last_score, v_loss, p_loss = agent.train()
    end = time.time()
    if solved:
        plt.plot(list(range(len(v_loss))), v_loss)
        plt.title("Value network's loss – Advantage Actor-Critic")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")

        plt.show()
        plt.plot(list(range(len(p_loss))), p_loss)
        plt.title("Policy network's loss – Advantage Actor-Critic")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.show()
        print(f"Scores - {curr_ep}, Time to converge - {end-start}")

    else:
        raise Exception(f"{env_name} FAILED")


def train_transfer(source_env, target_env):
    agent = ActorCriticAgent(env=gym.make(source_env))
    agent.set_env(source_env)

    solved, curr_ep, last_score, v_loss, p_loss, duration, ep_rew = agent.train()

    save_losses(curr_ep, v_loss, p_loss, f'{source_env}_source', label=f'{source_env} - Source', time=duration, ep_rew=ep_rew)

    if solved:
        agent.set_env(target_env)
        agent.init_output_weights()
        if target_env == 'MountainCarContinuous-v0':
            agent.set_params(learning_rate=0.00001, v_learning_rate=0.0005, discount_factor=0.999)

        solved, curr_ep, last_score, v_loss, p_loss, duration, ep_rew = agent.train()

        save_losses(curr_ep, v_loss, p_loss, f'{target_env}_target', label=f'{target_env} - Transfer', time=duration, ep_rew=ep_rew)
    else:
        raise Exception(f"{source_env} FAILED")


if __name__ == '__main__':
    default_env = 'mcc'
    if len(sys.argv) == 3:
        train_transfer(ENVS[sys.argv[1]], ENVS[sys.argv[2]])
    else:
        print("Wrong choice")
