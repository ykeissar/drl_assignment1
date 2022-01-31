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
        raise Exception("FAILED")


def train_transfer(source_env, target_env):#df=0.999, lr=0.0001, vlr=0.005
    agent = ActorCriticAgent(env=gym.make(source_env), render=False, hidden_layer_size=128, v_hidden_layer_size=64, discount_factor=0.999, learning_rate=0.0001, v_learning_rate=0.005)
    agent.set_env(source_env)

    solved, curr_ep, last_score, v_loss, p_loss, duration = agent.train()

    save_losses(curr_ep, v_loss, p_loss, f'{source_env}_source', label=f'{source_env} - Source', time=duration)

    if solved:
        agent.set_env(target_env)
        agent.init_output_weights()

        solved, curr_ep, last_score, v_loss, p_loss, duration = agent.train()

        save_losses(curr_ep, v_loss, p_loss, f'{target_env}_target', label=f'{target_env} - Target', time=duration)


if __name__ == '__main__':
    train_transfer(ENVS['acro'], ENVS['cp'])

    # train_transfer(ENVS['cp'], ENVS['mcc'])

    # compare between transferred cp to self-learned cp
    # insert json files
    # plot_losses([])

    # compare between transferred mcc to self-learned mcc
    # insert json files
    # plot_losses([])
