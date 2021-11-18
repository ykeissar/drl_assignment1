import numpy as np
import dqn_agent


def dqn(agent: dqn_agent.DQNAgent, episodes=5000, debug=False):
    ep_rewards = []
    for i in range(episodes):
        s = agent.env.reset()
        total_reward = 0
        done = False
        while not done:
            a = agent.sample_action(s.reshape(-1, 4))

            new_s, r, done, _ = agent.env.step(a)
            total_reward += r
            agent.experience_replay.append((s, a, r, new_s, done))

            agent.train_agent(done)
            s = new_s
            # if steps % 10 == 0:
            agent.env.render()

        print(f"Episode {i} reward:{total_reward}")

        # decaying e-greedy
        agent.epsilon /= 1.1

        ep_rewards.append(total_reward)

    return ep_rewards


