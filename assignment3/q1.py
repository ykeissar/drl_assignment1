import gym
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf

import sys
import logging as log
import time
from actor_critic_agent_ import ActorCriticAgent

np.random.seed(1)
N_ITERS = 10
tf.disable_v2_behavior()

ENV = 'CartPole-v1'


def only_one(env_name):
    agent = ActorCriticAgent(env=gym.make(env_name))

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


def multiple(n_iters):
    agent = ActorCriticAgent(env=gym.make('CartPole-v1'))

    times = []
    scores = []
    i = 0
    while i < n_iters:
        start = time.time()
        solved, curr_ep, _, _, _ = agent.train()
        end = time.time()
        if solved:
            scores.append(curr_ep)
            times.append(end-start)
            i += 1

    print(f"Scores mean - {np.mean(scores)}, Time to converge mean - {np.mean(times)}")


def tune_params():
    log.basicConfig(filename=f"log/q2_run_{time.strftime('%Y-%m-%d_%H:%M:%S', time.gmtime())}.log",
                    datefmt='%Y-%m-%d,%H:%M:%S', level=log.INFO)
    lrs = [0.0002, 0.0004, 0.0025, 0.005, 0.05]
    lrs_v = [0.0002, 0.0004, 0.0025, 0.005, 0.05]
    df = [0.999, 0.995, 0.99, 0.95]
    already_done = 30
    curr = 0
    for i in range(len(lrs)):
        for j in range(len(lrs_v)):
            for t in range(len(df)):
                if curr < already_done:
                    curr += 1
                    continue

                log.info(
                    f"START -- df={df[t]}, lr={lrs[i]}, v_lr={lrs_v[j]}")
                solved_all = True
                scores = []
                times = []
                for _ in range(3):
                    agent = ActorCriticAgent(env=gym.make('CartPole-v1'), discount_factor=df[t], learning_rate=lrs[i],
                                             v_learning_rate=lrs_v[j],)
                    start = time.time()
                    solved, curr_ep, _, _, _ = agent.train()
                    end = time.time()
                    if not solved:
                        solved_all = False
                        break
                    scores.append(curr_ep)
                    times.append(end - start)

                if solved_all:
                    log.info(
                        f"df={df[t]}, lr={lrs[i]}, vlr={lrs_v[j]} --- FINISHED - score - {np.mean(scores)} episodes, took:{np.mean(times)}")
                else:
                    log.info(
                        f"df={df[t]}, lr={lrs[i]}, vlr={lrs_v[j]} --- NOT FINISHED")

                curr += 1


if __name__ == '__main__':
    if len(sys.argv) < 2 or sys.argv[1] == '1':
        only_one(ENV)
        
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
        tune_params()
        
    else:
        print("Wrong choice")
            