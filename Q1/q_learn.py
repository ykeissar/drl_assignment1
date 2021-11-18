import numpy as np


def q_learn(env, Q=None, alpha=0.1, gamma=0.99, epsilon=1, episodes=5000, steps=100,
            debug=False):

    num_act = env.action_space.n
    num_states = env.observation_space.n
    if not Q:
        Q = np.zeros((num_states, num_act))

    ep_rewards = []
    ep_avg_steps = []
    ep100_steps_to_goal = []
    for i in range(episodes):
        s = env.reset()
        count_steps = 0
        count_rewards = 0

        for j in range(steps):
            count_steps += 1
            a = env.action_space.sample() if np.random.rand() < epsilon else np.argmax(Q[s, :])

            s_tag, r, done, _ = env.step(a)
            if done:
                if r > 0:
                    if debug:
                        print(f"done - episode {i} took - {count_steps} steps")
                    ep100_steps_to_goal.append(count_steps)
                    count_steps = 0
                    count_rewards += 1
                target = r
                s_tag = env.reset()
            else:
                target = r + gamma * max(Q[s_tag, :])
            Q[s, a] = (1 - alpha) * Q[s, a] + alpha * target
            s = s_tag

        if debug and np.sum(Q) > 0:
            print(Q)

        ep_rewards.append(count_rewards)

        # saves the avarage number of steps to goal in last 100 episodes
        if i % 100 == 99:
            if debug:
                print(f"Episode {i} - steps to goal -\n{ep100_steps_to_goal}")
            if ep100_steps_to_goal:
                ep_avg_steps.append(np.mean(ep100_steps_to_goal))
            else:
                ep_avg_steps.append(100)
            ep100_steps_to_goal = []

        # decaing e-greedy by 2 factor
        epsilon /= 1.1

    return Q, ep_rewards, ep_avg_steps