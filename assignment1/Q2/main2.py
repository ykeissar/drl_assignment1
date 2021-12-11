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

    plt.plot(list(range(len(ep_rewards_means))), ep_rewards_means)
    plt.xlabel('50 Episodes')
    plt.ylabel('Mean rewards')
    plt.show()


def tune_params():
    f = open('params2.txt', 'w+')

    env = gym.make('CartPole-v1')
    alphas =[]# [0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 1]
    gammas =[]# [0.9, 0.95, 0.99, 0.995, 0.999, 1]
    epsilon_decays = [0.9, 0.95, 0.99, 0.995, 0.999, 1]
    batch_size = [12, 24, 36, 48, 60]
    hidden_units = [16, 32, 48, 64]
    update_targets = [5, 20, 30, 50]
    epochs = [5, 20, 50, 100]
    
    e = 200
    e_to_calc = 30
    
    f.write('ALPHAS:\n')
    for i in range(len(alphas)):
        agent = DQNAgent(env,name=f'alpha:{alphas[i]}', lr=alphas[i])
        ep_rewards = dqn(agent, episodes=e)
        print(f"alpha:{alphas[i]}, rewards: {np.mean(ep_rewards[-e_to_calc:])}\n")
        f.write(f"alpha:{alphas[i]}, rewards: {np.mean(ep_rewards[-e_to_calc:])}\n")

    f.write('GAMMAS:\n')
    for i in range(len(gammas)):
        agent = DQNAgent(env, name=f'gamma:{gammas[i]}', gamma=gammas[i])
        ep_rewards = dqn(agent, episodes=e)
        print(f"gamma:{gammas[i]}, rewards: {np.mean(ep_rewards[-e_to_calc:])}\n")
        f.write(f"gamma:{gammas[i]}, rewards: {np.mean(ep_rewards[-e_to_calc:])}\n")

    f.write('DECAYS:\n')
    for i in range(len(epsilon_decays)):
        agent = DQNAgent(env, name=f'epsilon_decay:{epsilon_decays[i]}', epsilon_decay=epsilon_decays[i])
        ep_rewards = dqn(agent, episodes=e)
        print(f"epsilon_decay:{epsilon_decays[i]}, rewards: {np.mean(ep_rewards[-e_to_calc:])}\n")
        f.write(f"epsilon_decay:{epsilon_decays[i]}, rewards: {np.mean(ep_rewards[-e_to_calc:])}\n")

    f.write('BATCH_SIZES:\n')
    for i in range(len(batch_size)):
        agent = DQNAgent(env,f'batch_size:{batch_size[i]}', batch_size=batch_size[i])
        ep_rewards = dqn(agent, episodes=e)
        print(f"batch_size:{batch_size[i]}, rewards: {np.mean(ep_rewards[-e_to_calc:])}\n")
        f.write(f"batch_size:{batch_size[i]}, rewards: {np.mean(ep_rewards[-e_to_calc:])}\n")

    f.write('HIDDEN UNITS:\n')
    for i in range(len(hidden_units)):
        agent = DQNAgent(env,f'hidden_unit:{hidden_units[i]}', n_hidden_units=hidden_units[i])
        ep_rewards = dqn(agent, episodes=e)
        print(f"hidden_units:{hidden_units[i]}, rewards: {np.mean(ep_rewards[-e_to_calc:])}\n")
        f.write(f"hidden_units:{hidden_units[i]}, rewards: {np.mean(ep_rewards[-e_to_calc:])}\n")

    f.write('UPDATE TARGET:\n')
    for i in range(len(update_targets)):
        agent = DQNAgent(env,f'update_target:{update_targets[i]}', update_target_model=update_targets[i])
        ep_rewards = dqn(agent, episodes=e)
        print(f"update_targets:{update_targets[i]}, rewards: {np.mean(ep_rewards[-e_to_calc:])}\n")
        f.write(f"update_targets:{update_targets[i]}, rewards: {np.mean(ep_rewards[-e_to_calc:])}\n")

    f.write('UPDATE TARGET:\n')
    for i in range(len(update_targets)):
        agent = DQNAgent(env,f'update_target:{update_targets[i]}', update_target_model=update_targets[i])
        ep_rewards = dqn(agent, episodes=e)
        print(f"update_targets:{update_targets[i]}, rewards: {np.mean(ep_rewards[-e_to_calc:])}\n")
        f.write(f"update_targets:{update_targets[i]}, rewards: {np.mean(ep_rewards[-e_to_calc:])}\n")

    f.write('EPOCHS:\n')
    for i in range(len(epochs)):
        agent = DQNAgent(env,f'epochs:{epochs[i]}', epochs=epochs[i])
        ep_rewards = dqn(agent, episodes=e)
        print(f"epochs:{epochs[i]}, rewards: {np.mean(ep_rewards[-e_to_calc:])}\n")
        f.write(f"epochs:{epochs[i]}, rewards: {np.mean(ep_rewards[-e_to_calc:])}\n")


if __name__ == '__main__':
    main()
    # tune_params()
