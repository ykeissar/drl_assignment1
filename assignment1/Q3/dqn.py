import numpy as np
import dqn_agent


def dqn(agent: dqn_agent.DQNAgent, episodes=500, debug=False):
    ep_rewards = []
    for i in range(episodes):
        s = agent.env.reset()
        s = np.reshape(s, (-1, 4))
        total_reward = 0
        done = False
        while not done:
            a = agent.sample_action(s)
            new_s, r, done, _ = agent.env.step(a)
            new_s = np.reshape(new_s, (-1, 4))
            total_reward += r
            agent.add_to_replay(s, a, r, new_s, done)
            agent.train_agent()
            s = new_s
            # agent.env.render()
        if len(ep_rewards) > 100 and np.mean(ep_rewards[-100:]) > 475:
            print(f"DONE at ep - {i}")
            break
        print(f"Episode {i} reward:{total_reward}")
        ep_rewards.append(total_reward)
    return ep_rewards