import gym
import matplotlib.pyplot as plt
from q_learn import *


def main():
    env = gym.make('FrozenLake-v1')
    num_act = env.action_space.n
    num_states = env.observation_space.n

    Q, ep_rewards, ep_avg_steps = q_learn(env, epsilon=0.5)

    Q_500, _ = q_learn(env, episodes=5)
    print("Q after 500 steps:", Q_500, sep='\n')

    Q_full, ep_rewards, ep_avg_steps = q_learn(env)
    plt.plot(np.arange(5000), ep_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Rewards")
    plt.show()

    Q_2000, _ = q_learn(env, episodes=20)
    print("Q after 2000 steps:", Q_2000, sep='\n')
    plt.plot(np.arange(50)*100, ep_avg_steps)
    plt.xlabel("Episodes")
    plt.ylabel("Avg. steps to goal")
    plt.show()


if __name__ == '__main__':
    main()