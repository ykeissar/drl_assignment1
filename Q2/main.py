from dqn_agent import DQNAgent
from dqn import dqn
import gym
import numpy as np
import matplotlib.pyplot as plt


def main():
    env = gym.make('CartPole-v1')
    agent = DQNAgent(env)
    ep_rewards = dqn(agent, episodes=500)

    # rewards mean over each 50 consecutive episodes
    ep_rewards_means = np.mean(np.reshape(ep_rewards, (-1, 50)), axis=1)

    plt.plot(list(range(50, 501, 50)), ep_rewards_means)
    plt.xlabel('Episodes')
    plt.ylabel('Mean rewards over 50 episodes')
    plt.show()


if __name__ == '__main__':
    main()
