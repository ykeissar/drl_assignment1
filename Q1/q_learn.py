import numpy as np
import copy


def q_learn(env, Q=None, alpha=0.1, gamma=0.99, epsilon=0.99, ep_decay=1.07, lr_dec=0, episodes=5000, steps=100,
            debug=False):
    Qs = []
    num_act = env.action_space.n
    num_states = env.observation_space.n
    if not Q:
        Q = np.zeros((num_states, num_act))

    # holds the number of rewards for an episode
    total_ep_rewards = []
    # holds the number of average number of steps for an episode
    total_ep_avg_steps = []

    for i in range(episodes):
        s = env.reset()
        count_steps = 0
        count_rewards = 0
        ep_avg_steps = []

        for j in range(steps):
            count_steps += 1
            if np.random.rand() < epsilon:
                a = env.action_space.sample()
            else:
                a = rand_argmax(Q[s, :])

            s_tag, r, done, _ = env.step(a)
            # env.render()
            if done:
                if r > 0:
                    if debug:
                        print(f"done - episode {i} took - {count_steps} steps")
                    ep_avg_steps.append(count_steps)
                    count_steps = 0
                    count_rewards += 1
                target = r
                s_tag = env.reset()
            else:
                qm = max(Q[s_tag, :])
                target = r + gamma * qm
            Q[s, a] = (1 - alpha) * Q[s, a] + alpha * target

            s = s_tag

        total_ep_rewards.append(count_rewards)

        if ep_avg_steps:
            if debug:
                print(f"Episode {i} - steps to goal - {ep_avg_steps}")
            total_ep_avg_steps.append(np.mean(ep_avg_steps))
        else:
            total_ep_avg_steps.append(100)

        # decaying e-greedy epsilon
        if epsilon > 0.01:
            epsilon /= ep_decay

        if alpha > 0:
            alpha -= lr_dec
            alpha = 0 if alpha < 0 else alpha

        if i == 5 or i == 20:
            Qs.append(copy.deepcopy(Q))

    Qs.append(copy.deepcopy(Q))
    return Qs, total_ep_rewards, total_ep_avg_steps


def rand_argmax(arr):
    return np.random.choice(np.flatnonzero(arr == arr.max()))
