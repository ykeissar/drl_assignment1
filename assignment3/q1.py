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
    agent = ActorCriticAgent(env=gym.make(env_name), **get_args(env_name))
    agent.set_env(env_name)
    solved, curr_ep, last_score, v_loss, p_loss, duration, ep_rew = agent.train()
    if solved:
        save_losses(curr_ep, v_loss, p_loss, f'{env_name}', label=f'{env_name}', time=duration, ep_rew=ep_rew)
    else:
        raise Exception("FAILED")


def tune_params(env_name):
    log.basicConfig(filename=f"log/tune_params_{env_name}_{time.strftime('%Y-%m-%d_%H:%M:%S', time.gmtime())}.log",
                    datefmt='%Y-%m-%d,%H:%M:%S', level=log.INFO,
                    format='%(asctime)s.%(msecs)03d %(levelname)s {%(module)s} [%(funcName)s] %(message)s')

    log.info(f"Tune params for {env_name}")
    lrs = [0.00001, 0.00005, 0.0001, 0.0002, 0.0005]
    lrs_v = [0.0005, 0.005]
    df = [0.999, 0.99]
    # sizes = [128]
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
                # for _ in range(1):
                agent = ActorCriticAgent(env=gym.make(env_name), discount_factor=df[t], learning_rate=lrs[i],
                                         v_learning_rate=lrs_v[j], max_episodes=500)
                agent.set_env(env_name)
                solved, curr_ep, _, _, _, duration, ep_rew = agent.train()
                # if not solved:
                #     solved_all = False
                #     tf.reset_default_graph()
                #     break
                tf.reset_default_graph()
                scores.append(curr_ep)
                times.append(duration)

                if solved:
                    log.info(
                        f"df={df[t]}, lr={lrs[i]}, vlr={lrs_v[j]}--- FINISHED - score - {np.mean(scores)} episodes, took:{np.mean(times)}\nlast 20{ep_rew[-20:]}\ntop 20:{sorted(ep_rew)[-20:]}")
                else:
                    log.info(
                        f"df={df[t]}, lr={lrs[i]}, vlr={lrs_v[j]} --- NOT FINISHED, \nlast 20{ep_rew[-20:]}\ntop 20:{sorted(ep_rew)[-20:]}")

                curr += 1


if __name__ == '__main__':
    default_env = 'mcc'
    if len(sys.argv) == 2:
        only_one(env_name=ENVS[sys.argv[1]])
    else:
        print("Wrong choice")
            