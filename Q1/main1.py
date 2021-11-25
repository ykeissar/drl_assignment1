import gym
import matplotlib.pyplot as plt
from q_learn import *


def main():
    env = gym.make('FrozenLake-v1')
    Qs, total_ep_rewards, total_ep_avg_steps = q_learn(env)

    plt.imshow(Qs[0])
    plt.colorbar()
    plt.title("Q after 500 steps")
    plt.show()
    print(f"Q500-\n{Qs[0]}")

    plt.imshow(Qs[1])
    plt.colorbar()
    plt.title("Q after 2000 steps")
    plt.show()
    print(f"Q2000-\n{Qs[1]}")

    plt.imshow(Qs[2])
    plt.colorbar()
    plt.title("Q final state")
    plt.show()
    print(f"Q final-\n{Qs[2]}")

    plt.plot(np.arange(len(total_ep_rewards)), total_ep_rewards)
    plt.xlabel("Episodes")
    plt.ylabel("Rewards")
    plt.title("Reward per episode")
    plt.show()

    mean_ep_avg_steps = np.mean(np.array(total_ep_avg_steps).reshape(-1, 100), axis=1)
    plt.plot(np.arange(len(mean_ep_avg_steps)), mean_ep_avg_steps)
    plt.xlabel("100 Episodes")
    plt.ylabel("Average steps to goal")
    plt.title("Avg Steps to Goal of 100 Episodes")
    plt.show()


def plot_results(ep_rewards, ep_avg_steps, title=''):
    ep_rewards = np.mean(np.array(ep_rewards).reshape(-1, 50), axis=1)
    ep_avg_steps = np.mean(np.array(ep_avg_steps).reshape(-1, 50), axis=1)

    n_episodes = len(ep_rewards)

    plt.plot(np.arange(n_episodes), ep_rewards, scaley=False)
    plt.xlabel("50 Episodes")
    plt.ylabel("Rewards")
    plt.title(title)
    ax = plt.gca()
    ax.set_ylim([0, 4])
    plt.show()

    plt.plot(np.arange(n_episodes), ep_avg_steps, scaley=False)
    plt.xlabel("50 Episodes")
    plt.ylabel("Average of steps to goal")
    plt.title(title)
    ax = plt.gca()
    ax.set_ylim([0, 100])
    plt.show()


def grid_tune_params():
    # 0.05, 0.1, 0.15,..., 1
    alphas = np.array(range(1, 11, 1))/10
    # decays = [1, 1.01, 1.03, 1.05, 1.07, 1.1]
    rew_scores = []
    steps_scores = []

    env = gym.make('FrozenLake-v1')

    for i in range(len(alphas)):
        Q, total_ep_rewards, total_ep_steps = q_learn(env, alpha=alphas[i])

        rew_scores.append(np.mean(total_ep_rewards))
        steps_scores.append(np.mean(total_ep_steps))

    print(rew_scores)
    print(steps_scores)


if __name__ == '__main__':
    main()
    # grid_tune_params()
