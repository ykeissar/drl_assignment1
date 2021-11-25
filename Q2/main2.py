from dqn_agent import DQNAgent
from dqn import dqn
import gym
import numpy as np
import matplotlib.pyplot as plt


def main():
    env = gym.make('CartPole-v1')
    agent = DQNAgent(env, tb=False)
    ep_rewards = dqn(agent)
    # rewards mean over each 50 consecutive episodes
    ep_rewards_means = np.mean(np.reshape(ep_rewards, (-1, 50)), axis=1)

    plt.plot(list(range(len(ep_rewards_means))), ep_rewards_means)
    plt.xlabel('50 Episodes')
    plt.ylabel('Mean rewards')
    plt.show()


def tune_params():
    f = open('params.txt', 'w')

    env = gym.make('CartPole-v1')
    alphas = [0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 1]
    gammas = [0.9, 0.95, 0.99, 0.995, 0.999, 1]
    epsilon_decays = [0.9, 0.95, 0.99, 0.995, 0.999, 1]
    batch_size = [12, 24, 36, 48, 60]
    
    e = 60
    e_to_calc = 35
    
    f.write('ALPHAS:\n')
    for i in range(len(alphas)):
        agent = DQNAgent(env, lr=alphas[i], tb=False)
        ep_rewards = dqn(agent, episodes=e)
        f.write(f"alpha:{alphas[i]}, rewards: {np.mean(ep_rewards[e_to_calc:])}\n")

    f.write('GAMMAS:\n')
    for i in range(len(gammas)):
        agent = DQNAgent(env, gamma=gammas[i], tb=False)
        ep_rewards = dqn(agent, episodes=e)
        f.write(f"gamma:{gammas[i]}, rewards: {np.mean(ep_rewards[e_to_calc:])}\n")

    f.write('DECAYS:\n')
    for i in range(len(epsilon_decays)):
        agent = DQNAgent(env, epsilon_decay=epsilon_decays[i], tb=False)
        ep_rewards = dqn(agent, episodes=e)
        f.write(f"epsilon_decay:{epsilon_decays[i]}, rewards: {np.mean(ep_rewards[e_to_calc:])}\n")

    f.write('BATCH_SIZES:\n')
    for i in range(len(batch_size)):
        agent = DQNAgent(env, batch_size=batch_size[i], tb=False)
        ep_rewards = dqn(agent, episodes=e)
        f.write(f"batch_size:{batch_size[i]}, rewards: {np.mean(ep_rewards[e_to_calc:])}\n")


if __name__ == '__main__':
    # main()
    tune_params()
