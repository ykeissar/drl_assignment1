import gym
import tensorflow.compat.v1 as tf
import collections
import numpy as np
from actor_critic_agent_ import *
import time

tf.disable_v2_behavior()


class PNNAgent:
    def __init__(self, env=None, discount_factor=0.999, learning_rate=0.0002, v_learning_rate=0.005, render=False,
                 state_size=4, max_episodes=5000, max_steps=501, source_networks=None, target_network=None):
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.v_learning_rate = v_learning_rate
        self.render = render

        self.env = env
        self.state_size = env.observation_space.n if hasattr(env.observation_space, 'n') else env.observation_space.shape[0]
        self.action_size = env.action_space.n if hasattr(env.action_space, 'n') else env.action_space.shape[0]

        self.max_episodes = max_episodes
        self.max_steps = max_steps

        self.padded_state_size = 6
        self.padded_action_size = 10

        tf.reset_default_graph()
        self.policy = ProgressiveNeuralNetwork(source_networks=source_networks, target_network=target_network)
        self.value = ValueNetwork(self.padded_state_size, self.v_learning_rate, 64)

        self.curr_env_name = ''

    def train(self, episodes=None):
        start = time.time()
        if episodes == None:
            episodes = self.max_episodes
        # Start training the agent with REINFORCE algorithm
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            solved = False
            Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])
            episode_rewards = np.zeros(episodes)
            average_rewards = 0.0
            v_loss = []
            p_loss = []
            for episode in range(episodes):
                state = self.env.reset()
                state = state.reshape([1, self.state_size])

                I = 1
                for step in range(self.max_steps):
                    actions_distribution = self.policy.predict(state, sess)
                    action = self.get_action(actions_distribution)
                    next_state, reward, done, _ = self.take_step(action)
                    next_state = next_state.reshape([1, self.state_size])
                    episode_rewards[episode] += reward
                    if self.render:
                        self.env.render()

                    if done:
                        v_s_tag = 0
                    else:
                        v_s_tag = self.value.predict(next_state, sess)

                    v_s = self.value.predict(state, sess)
                    delta = reward + self.discount_factor * v_s_tag - v_s

                    if self.curr_env_name == 'MountainCarContinuous-v0' and False:
                        act = actions_distribution
                    else:
                        act = np.zeros(self.action_size)
                        act[action] = 1

                    _, loss = self.value.update(state, reward + self.discount_factor*v_s_tag, sess)
                    v_loss.append(loss)
                    _, loss = self.policy.update(state, I * delta, act, sess)
                    p_loss.append(loss)
                    if done:
                        break

                    I *= self.discount_factor
                    state = next_state

                if episode > 98:
                    # Check if solved
                    average_rewards = np.mean(episode_rewards[(episode - 99):episode + 1])
                print(f"Episode {episode} Reward: {episode_rewards[episode]} Average over 100 episodes: {round(average_rewards, 2)}")
                if episode > 98 and self.goal_reached(average_rewards):
                    print(' Solved at episode: ' + str(episode))
                    solved = True

                if solved:
                    end = time.time()
                    return solved, episode, average_rewards, v_loss, p_loss, end-start
        end = time.time()
        return False, -1, average_rewards, [], [], end-start

    def get_action(self, actions_distribution):
        actions_distribution[self.action_size:] = 0
        if self.curr_env_name == 'MountainCarContinuous-v0' and False:
            return actions_distribution
        sum_p = sum(actions_distribution)
        actions_distribution = [i / sum_p for i in actions_distribution]

        return np.random.choice(np.arange(len(actions_distribution)), p=actions_distribution)

    def goal_reached(self, average_rewards):
        if self.curr_env_name == 'CartPole-v1':
            return average_rewards > 475
        if self.curr_env_name == 'Acrobot-v1':
            return average_rewards > -81
        if self.curr_env_name == 'MountainCarContinuous-v0':
            return average_rewards > 0

    def set_env(self, env_name):
        self.curr_env_name = env_name
        self.policy.set_env(env_name)
        self.env = gym.make(env_name)

        if env_name == 'MountainCarContinuous-v0':
            self.action_size = self.padded_action_size
        else:
            self.action_size = self.env.action_space.n if hasattr(self.env.action_space, 'n') else self.env.action_space.shape[0]

        self.state_size = self.env.observation_space.n if hasattr(self.env.observation_space, 'n') else self.env.observation_space.shape[0]

    def take_step(self, action):
        act = action
        if self.curr_env_name == 'MountainCarContinuous-v0':
            # discretesizing action, from [0:self.action_size] number to corelating [-1:1] value
            act = [round(action / ((self.action_size - 1) / 2) - 1, 2)]
        return self.env.step(act)

    def init_output_weights(self):
        self.policy.init_output_weights()

    def freeze_source(self):
        # Freeze all but U2 in source network
        self.policy.freeze_source()

    def unfreeze_source(self):
        # Unfreeze all in source network
        self.policy.unfreeze_source()


class ProgressiveNeuralNetwork:
    def __init__(self, source_networks, target_network, a=1): #) state_size, action_size, learning_rate, name='policy_network'):
        self.target_network = target_network
        self.source_networks = source_networks
        self.a = a

        with tf.variable_scope('policy_network'):
            self.U2 = tf.get_variable("U2", [12, self.target_network.action_size], initializer=tf.keras.initializers.glorot_normal(seed=0))

            # connect target and source columns
            self.target_network.output = tf.add(tf.add(tf.matmul(self.target_network.A1, self.target_network.W2), self.target_network.b2),
                                                tf.multiply(self.a, tf.matmul(self.source_network.A1, self.U2)))

    def predict(self, state, sess):
        return sess.run([self.target_network.actions_distribution, self.source_network.actions_distribution],
                        {self.target_network.state: pad(state, self.target_network.state_size),
                         self.source_network.state: pad(state, self.source_network.state_sizee)})

    def update(self, state, delta, action, sess):
        feed_dict = {self.source_network.state: pad(state, self.source_network.state_size), self.source_network.delta: delta, self.source_network.action: pad(action, self.source_network.action_size),
                     self.target_network.state: pad(state, self.target_network.state_size), self.target_network.delta: delta, self.target_network.action: pad(action, self.target_network.action_size)}

        return sess.run([self.source_network.optimizer, self.source_network.loss,self.target_network.optimizer, self.target_network.loss], feed_dict)

    def freeze_source(self):
        # Freeze all but U2 in source network
        self.source_network.optimizer = tf.train.AdamOptimizer(learning_rate=self.source_network.learning_rate) \
            .minimize(self.source_network.loss, var_list=['U2'])

    def unfreeze_source(self):
        # Unfreeze all in source network
        self.source_network.optimizer = tf.train.AdamOptimizer(learning_rate=self.source_network.learning_rate) \
            .minimize(self.source_network.loss)

    def init_output_weights(self):
        tf.variables_initializer(['W2', 'b2'])
