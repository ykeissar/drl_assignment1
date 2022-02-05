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
    agent = PNNAgent(env=gym.make(target_env_name), source_networks=sources, target_network=target, max_episodes=1500)
    agent.set_env(target_env_name)

    solved, curr_ep, last_score, v_loss, p_loss, duration, ep_rew = agent.train()

    save_losses(curr_ep, v_loss, p_loss, f'{target_env_name}_target_pnn', label=f'{target_env_name} - Target - PNN',
                time=duration, ep_rew=ep_rew)


if __name__ == '__main__':
    if len(sys.argv) == 4:

        sources = [sys.argv[1], sys.argv[2]]
        target = sys.argv[3]

        agent0 = ActorCriticAgent(env=gym.make(ENVS[sources[0]]), var_pref='0_', **get_args(sources[0]))
        agent0.set_env(ENVS[sources[0]])
        solved0, _, _, _, _, _, _ = agent0.train()

        agent1 = ActorCriticAgent(env=gym.make(ENVS[sources[1]]), var_pref='1_', **get_args(sources[1]))
        agent1.set_env(ENVS[sources[1]])

        solved1, _, _, _, _, _, _ = agent1.train()

        if solved0 and solved1:
            agent_t = ActorCriticAgent(env=gym.make(ENVS[target]), var_pref='t_', **get_args(target))
            agent_t.set_env(ENVS[target])

            run_pnn(ENVS[target], [agent0.policy, agent1.policy], agent_t.policy)

        else:
            print(f"{', '.join([sources[i] for i,s in enumerate([solved0, solved1]) if s])} FAILED")
    else:
        print("Wrong choice")
