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
    solved, curr_ep, last_score, v_loss, p_loss, duration = agent.train()
    if solved:
        save_losses(curr_ep, v_loss, p_loss, f'{env_name}', label=f'{env_name} - Source', time=duration)
    else:
        raise Exception("FAILED")


def multiple(n_iters):
    agent = ActorCriticAgent(env=gym.make('CartPole-v1'))

    times = []
    scores = []
    i = 0
    while i < n_iters:
        solved, curr_ep, _, _, _, duration = agent.train()
        if solved:
            scores.append(curr_ep)
            times.append(duration)
            i += 1

    print(f"Scores mean - {np.mean(scores)}, Time to converge mean - {np.mean(times)}")


def tune_params(env_name):
    log.basicConfig(filename=f"log/tune_params_{env_name}_{time.strftime('%Y-%m-%d_%H:%M:%S', time.gmtime())}.log",
                    datefmt='%Y-%m-%d,%H:%M:%S', level=log.INFO,
                    format='%(asctime)s.%(msecs)03d %(levelname)s {%(module)s} [%(funcName)s] %(message)s')

    log.info(f"Tune params for {env_name}")
    lrs = [0.0001, 0.0005, 0.001, 0.005]
    lrs_v = [0.0001, 0.0005, 0.001, 0.005]
    df = [0.999, 0.99, 0.95]
    already_done = 0
    curr = 0
    for i in range(len(lrs)):
        for j in range(len(lrs_v)):
            for t in range(len(df)):
                if curr < already_done:
                    curr += 1
                    continue

                log.info(f"START -- df={df[t]}, lr={lrs[i]}, v_lr={lrs_v[j]}")
                solved_all = True
                scores = []
                times = []
                for _ in range(3):
                    agent = ActorCriticAgent(env=gym.make(env_name), discount_factor=df[t], learning_rate=lrs[i],
                                             v_learning_rate=lrs_v[j], max_episodes=3000, hidden_layer_size=128, v_hidden_layer_size=64)
                    agent.set_env(env_name)
                    solved, curr_ep, _, _, _, duration = agent.train()
                    if not solved:
                        solved_all = False
                        break
                    scores.append(curr_ep)
                    times.append(duration)

                if solved_all:
                    log.info(
                        f"df={df[t]}, lr={lrs[i]}, vlr={lrs_v[j]} --- FINISHED - score - {np.mean(scores)} episodes, took:{np.mean(times)}")
                else:
                    log.info(
                        f"df={df[t]}, lr={lrs[i]}, vlr={lrs_v[j]} --- NOT FINISHED")

                curr += 1


if __name__ == '__main__':
    if len(sys.argv) < 2 or sys.argv[1] == '1':
        only_one(ENVS['mcc'])
        
    elif sys.argv[1] == '2':
        if len(sys.argv) > 2:
            multiple(int(sys.argv[2]))
        else:
            multiple(N_ITERS)
    
    elif sys.argv[1] == '2':
        if len(sys.argv) > 2:
            multiple(int(sys.argv[2]))
        else:
            multiple(N_ITERS)
    elif sys.argv[1] == '3':
        if len(sys.argv) > 2:
            tune_params(env_name=ENVS[sys.argv[2]])
        else:
            tune_params(env_name=ENVS['cp'])
    else:
        print("Wrong choice")
            