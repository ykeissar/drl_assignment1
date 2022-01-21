import gym
import numpy as np
import logging as log
import time
import collections
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
import sys, json


tf.disable_v2_behavior()


env = gym.make('CartPole-v1')


np.random.seed(1)


class PolicyNetwork:
    def __init__(self, state_size, action_size, learning_rate, name='policy_network'):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate

        with tf.variable_scope(name):
            self.state = tf.placeholder(tf.float32, [None, self.state_size], name="state")
            self.action = tf.placeholder(tf.int32, [self.action_size], name="action")
            self.delta = tf.placeholder(tf.float32, name="delta")

            self.W1 = tf.get_variable("W1", [self.state_size, 12], initializer=tf.keras.initializers.glorot_normal(seed=0))
            self.b1 = tf.get_variable("b1", [12], initializer=tf.zeros_initializer())
            self.W2 = tf.get_variable("W2", [12, self.action_size], initializer=tf.keras.initializers.glorot_normal(seed=0))
            self.b2 = tf.get_variable("b2", [self.action_size], initializer=tf.zeros_initializer())
            # self.W3 = tf.get_variable("W3", [12, self.action_size], initializer=tf.keras.initializers.glorot_normal(seed=0))
            # self.b3 = tf.get_variable("b3", [self.action_size], initializer=tf.zeros_initializer())

            self.Z1 = tf.add(tf.matmul(self.state, self.W1), self.b1)
            self.A1 = tf.nn.relu(self.Z1)
            # self.Z2 = tf.add(tf.matmul(self.A1, self.W2), self.b2)
            # self.A2 = tf.nn.relu(self.Z2)
            self.output = tf.add(tf.matmul(self.A1, self.W2), self.b2)

            # Softmax probability distribution over actions
            self.actions_distribution = tf.squeeze(tf.nn.softmax(self.output))
            # Loss with negative log probability
            self.neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.output, labels=self.action)
            self.loss = tf.reduce_mean(self.neg_log_prob * self.delta)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def predict(self, state, sess):
        return sess.run(self.actions_distribution, {self.state: state})

    def update(self, state, delta, action, sess):
        feed_dict = {self.state: state, self.delta: delta, self.action: action}
        return sess.run([self.optimizer, self.loss], feed_dict)


class ValueNetwork:
    def __init__(self, state_size, learning_rate, name='value_network'):
        self.state_size = state_size
        self.learning_rate = learning_rate

        with tf.variable_scope(name):
            self.state = tf.placeholder(tf.float32, [None, self.state_size], name="v_state")
            self.delta = tf.placeholder(tf.float32, name="delta")
            self.target = tf.placeholder(tf.float32, name="target")

            self.W1 = tf.get_variable("v_W1", [self.state_size, 12], initializer=tf.keras.initializers.glorot_normal(seed=0))
            self.b1 = tf.get_variable("v_b1", [12], initializer=tf.zeros_initializer())
            self.W2 = tf.get_variable("v_W2", [12, 1], initializer=tf.keras.initializers.glorot_normal(seed=0))
            self.b2 = tf.get_variable("v_b2", [1], initializer=tf.zeros_initializer())
            # self.W3 = tf.get_variable("v_W3", [12, 1], initializer=tf.keras.initializers.glorot_normal(seed=0))
            # self.b3 = tf.get_variable("v_b3", [1], initializer=tf.zeros_initializer())

            self.Z1 = tf.add(tf.matmul(self.state, self.W1), self.b1)
            self.A1 = tf.nn.relu(self.Z1)
            # self.Z2 = tf.add(tf.matmul(self.A1, self.W2), self.b2)
            # self.A2 = tf.nn.relu(self.Z2)
            self.output = tf.add(tf.matmul(self.A1, self.W2), self.b2)
            # self.W = np.random.normal(size=(1,self.state))

            # Softmax probability distribution over actions
            self.v_s = tf.squeeze(self.output)
            # Loss with negative log probability
            self.loss = tf.squared_difference(self.v_s, self.target)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def predict(self, state, sess):
        return sess.run(self.v_s, {self.state: state})
        # return self.W @ state

    def update(self, state, target, sess):
        feed_dict = {self.state: state, self.target: target}
        return sess.run([self.optimizer, self.loss], feed_dict)


def run_pg(discount_factor=0.99, learning_rate=0.0004, v_learning_rate=0.0002, render=False):
    # Define hyperparameters
    state_size = 4
    action_size = env.action_space.n

    max_episodes = 2000
    max_steps = 501

    # Initialize the policy network
    tf.reset_default_graph()
    policy = PolicyNetwork(state_size, action_size, learning_rate)
    value = ValueNetwork(state_size, v_learning_rate)

    # Start training the agent with REINFORCE algorithm
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        solved = False
        Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])
        episode_rewards = np.zeros(max_episodes)
        average_rewards = 0.0
        v_loss = []
        p_loss = []
        for episode in range(max_episodes):
            state = env.reset()
            state = state.reshape([1, state_size])
            episode_transitions = []

            for step in range(max_steps):
                actions_distribution = policy.predict(state, sess)
                action = np.random.choice(np.arange(len(actions_distribution)), p=actions_distribution)
                next_state, reward, done, _ = env.step(action)
                next_state = next_state.reshape([1, state_size])

                if render:
                    env.render()

                action_one_hot = np.zeros(action_size)
                action_one_hot[action] = 1
                episode_transitions.append(
                    Transition(state=state, action=action_one_hot, reward=reward, next_state=next_state, done=done))
                episode_rewards[episode] += reward

                if done:
                    if episode > 98:
                        # Check if solved
                        average_rewards = np.mean(episode_rewards[(episode - 99):episode + 1])
                    print(
                        "Episode {} Reward: {} Average over 100 episodes: {}".format(episode, episode_rewards[episode],
                                                                                     round(average_rewards, 2)))
                    if average_rewards > 475:
                        print(' Solved at episode: ' + str(episode))
                        solved = True
                    break
                state = next_state

            if solved:
                return solved, episode, average_rewards, v_loss, p_loss

            # Compute Rt for each time-step t and update the network's weights
            for t, transition in enumerate(episode_transitions):
                total_discounted_return = sum(
                    discount_factor ** i * tr.reward for i, tr in enumerate(episode_transitions[t:]))  # Rt

                base_line = value.predict(transition.state, sess)
                delta = total_discounted_return - base_line

                _, loss = value.update(transition.state, delta, sess)
                v_loss.append(loss)

                _, loss = policy.update(transition.state, delta, transition.action, sess)
                p_loss.append(loss)

    return False, -1, average_rewards, [], []

N_ITERS = 20


def only_one():
    start = time.time()
    solved, curr_ep, last_score, v_loss, p_loss = run_pg()
    end = time.time()
    if solved:
        losses_p = [str(p) for p in p_loss]
        with open('losses/rein_p.json', 'w+') as f:
            json.dump(losses_p, f)
        losses_v = [str(p) for p in v_loss]

        with open('losses/rein_v.json', 'w+') as f:
            json.dump(losses_v,f)

        plt.plot(list(range(len(v_loss))), v_loss)
        plt.title("Value network's loss – Advantage Actor-Critic")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")

        plt.show()
        plt.plot(list(range(len(p_loss))), p_loss)
        plt.title("Policy network's loss – Advantage Actor-Critic")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.show()
        print(f"Scores - {curr_ep}, Time to converge - {end-start}")

    else:
        raise Exception("FAILED")


def multiple(n_iters):
    times = []
    scores = []
    for _ in range(n_iters):
        start = time.time()
        solved, curr_ep, _, _, _ = run_pg()
        end = time.time()
        if solved:
            scores.append(curr_ep)
            times.append(end-start)
        else:
            raise Exception("FAILED")

    print(f"Scores mean - {np.mean(scores)}, Time to converge mean - {np.mean(times)}")


def tune_params():
    log.basicConfig(filename=f"log/q1_run_{time.strftime('%Y-%m-%d_%H:%M:%S', time.gmtime())}.log",
                    datefmt='%Y-%m-%d,%H:%M:%S', level=log.INFO)
    lrs = [0.0002, 0.0004, 0.0025, 0.005, 0.05]
    lrs_v = [0.0002, 0.0004, 0.0025, 0.005, 0.05]
    df = [0.999, 0.995, 0.99, 0.95]
    already_done = 75
    curr = 0
    for i in range(len(lrs)):
        for j in range(len(lrs_v)):
            for t in range(len(df)):
                if curr < already_done:
                    curr += 1
                    continue

                log.info(
                    f"START -- df={df[t]}, lr={lrs[i]}, v_lr={lrs_v[j]}")
                solved_all = True
                scores = []
                times = []
                for _ in range(3):
                    start = time.time()
                    solved, curr_ep, _, _, _ = run_pg(discount_factor=df[t], learning_rate=lrs[i], v_learning_rate=lrs_v[j],
                                                render=False)
                    end = time.time()
                    if not solved:
                        solved_all = False
                        break
                    scores.append(curr_ep)
                    times.append(end - start)

                if solved_all:
                    log.info(
                        f"df={df[t]}, lr={lrs[i]}, vlr={lrs_v[j]} --- FINISHED - score - {np.mean(scores)} episodes, took:{np.mean(times)}")
                else:
                    log.info(
                        f"df={df[t]}, lr={lrs[i]}, vlr={lrs_v[j]} --- NOT FINISHED")

                curr += 1


if __name__ == '__main__':
    if len(sys.argv) < 2 or sys.argv[1] == '1':
        only_one()

    elif sys.argv[1] == '2':
        if len(sys.argv) > 2:
            multiple(int(sys.argv[2]))
        else:
            multiple(N_ITERS)

    elif sys.argv[1] == '2':
        if len(sys.argv) > 2:
            multiple(int(sys.argv[2]))
        else:
            multiple(N_ITERS)
    elif sys.argv[1] == '3':
        tune_params()

    else:
        print("Wrong choice")
        