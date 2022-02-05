import tensorflow.compat.v1 as tf
from actor_critic_agent_ import *
import time

tf.disable_v2_behavior()


class PNNAgent:
    def __init__(self, env=None, discount_factor=0.999, learning_rate=0.0002, v_learning_rate=0.005, render=False,
                 epsilon=1, max_episodes=5000, max_steps=501, source_networks=None, target_network=None):
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

        # tf.reset_default_graph()
        self.policy = ProgressiveNeuralNetwork(source_networks=source_networks, target_network=target_network)
        self.value = ValueNetwork(self.padded_state_size, self.v_learning_rate, 64)

        self.curr_env_name = ''
        self.epsilon = epsilon

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
                    try:
                        action = self.get_action(actions_distribution,  sess)
                    except Exception as e:
                        logging.error(e)
                        continue
                    next_state, reward, done, _ = self.take_step(action)
                    next_state = next_state.reshape([1, self.state_size])
                    episode_rewards[episode] += reward

                    if self.curr_env_name == 'MountainCarContinuous-v0':
                        position = state[0][0]
                        reward += (position if position > 0 else -position)

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

                # if episode > 98:
                #     # Check if solved
                #     average_rewards = np.mean(episode_rewards[(episode - 99):episode + 1])
                # print(f"Episode {episode} Reward: {episode_rewards[episode]} Average over 100 episodes: {round(average_rewards, 2)}")
                # if episode > 98 and self.goal_reached(average_rewards):
                #     print(' Solved at episode: ' + str(episode))
                #     solved = True
                #
                # if solved:
                #     end = time.time()
                #     return solved, episode, average_rewards, v_loss, p_loss, end-start

                if episode > 98:
                    # Check if solved
                    average_rewards = np.mean(episode_rewards[(episode - 99):episode + 1])
                print(f"{self.curr_env_name} - Episode {episode} Reward: {episode_rewards[episode]} Average over 100 episodes: {round(average_rewards, 2)}")
                if self.goal_reached(episode, curr_episode_rewards=episode_rewards[((episode - 99) if episode > 98 else 0):episode + 1]):
                    print(f'{self.curr_env_name} Solved at episode: ' + str(episode))
                    solved = True
                if self.hopeless(average_rewards, episode):
                    break

                if solved:
                    end = time.time()
                    return solved, episode, average_rewards, v_loss, p_loss, end-start, episode_rewards

                self.epsilon *= 0.99

        end = time.time()
        return False, episode, average_rewards, v_loss, p_loss, end-start, episode_rewards

    def hopeless(self, average_rewards, episode):
        if self.curr_env_name == 'CartPole-v1':
            return episode > 700 and average_rewards < 13
        if self.curr_env_name == 'Acrobot-v1':
            return average_rewards < -495
        if self.curr_env_name == 'MountainCarContinuous-v0':
            return False

    def get_action(self, actions_distribution, sess):
        actions_distribution[self.action_size:] = 0

        if self.curr_env_name == 'MountainCarContinuous-v0' and not DISCRETIZATION:
            if np.random.rand() < self.epsilon:
                return [np.random.uniform(-1, 1)]

            mu = actions_distribution[0]
            sigma = np.power(actions_distribution[1], 2, dtype=np.float32)
            self.policy.dist = tfd.Normal(loc=mu, scale=sigma)

            act = self.policy.dist.sample()
            act = tf.tanh(act)
            act = sess.run(act)
            return [act]

        actions_distribution = [ac + (1e-15) if i < self.action_size else ac for i, ac in enumerate(actions_distribution)]
        sum_p = sum(actions_distribution)

        actions_distribution = [i / sum_p for i in actions_distribution]
        try:
            a = np.random.choice(np.arange(len(actions_distribution)), p=actions_distribution)
        except Exception as e:
            raise e
        return a

    def goal_reached(self, episode, curr_episode_rewards):
        if episode > 98 and self.curr_env_name == 'CartPole-v1':
            return np.mean(curr_episode_rewards) > 475
        if episode > 98 and self.curr_env_name == 'Acrobot-v1':
            return np.mean(curr_episode_rewards) > -90
        if self.curr_env_name == 'MountainCarContinuous-v0':
            return len([r for r in curr_episode_rewards if r > 0]) > 9
        return False

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

    def set_params(self, learning_rate, v_learning_rate, discount_factor):
        if self.curr_env_name == 'MountainCarContinuous-v0':
            self.policy.learning_rate = learning_rate
            self.policy.v_learning_rate = v_learning_rate
            self.policy.discount_factor = discount_factor

    def set_env(self, env_name):
        self.curr_env_name = env_name
        self.policy.set_env(env_name)
        self.value.set_env(env_name)
        self.env = gym.make(env_name)

        if env_name == 'MountainCarContinuous-v0':
            self.action_size = self.padded_action_size
        else:
            self.action_size = self.env.action_space.n if hasattr(self.env.action_space, 'n') else self.env.action_space.shape[0]

        self.state_size = self.env.observation_space.n if hasattr(self.env.observation_space, 'n') else self.env.observation_space.shape[0]



class ProgressiveNeuralNetwork:
    def __init__(self, source_networks, target_network): #) state_size, action_size, learning_rate, name='policy_network'):
        self.target_network = target_network
        self.source_networks = source_networks
        self.scaler = MinMaxScaler().fit(np.array([-1.2, -0.07, 0.6, 0.07]).reshape((-1, 2)))

        with tf.variable_scope('pnn'):
            self.U1 = tf.get_variable("U1", [self.source_networks[0].hidden_layer_size, self.target_network.action_size], initializer=tf.keras.initializers.glorot_normal(seed=0))
            self.U2 = tf.get_variable("U2", [self.source_networks[1].hidden_layer_size, self.target_network.action_size], initializer=tf.keras.initializers.glorot_normal(seed=0))

            # connect target and source columns
            self.target_network.output = tf.add(tf.add(tf.add(tf.matmul(self.target_network.A1, self.target_network.W2), self.target_network.b2),
                                                tf.matmul(self.source_networks[0].A1, self.U1)),
                                                tf.matmul(self.source_networks[1].A1, self.U2))

    def predict(self, state, sess):
        # predict by calculating all actions_distributions, returning only target's one.
        return sess.run([self.target_network.actions_distribution,
                         self.source_networks[0].actions_distribution,
                         self.source_networks[1].actions_distribution],
                        {self.target_network.state: pad(self.process(state), self.target_network.state_size),
                         self.source_networks[0].state: pad(self.process(state), self.source_networks[0].state_size),
                         self.source_networks[1].state: pad(self.process(state), self.source_networks[1].state_size)})[0]

    def update(self, state, delta, action, sess):

        feed_dict = {self.target_network.state: pad(self.process(state), self.target_network.state_size),
                     self.target_network.delta: delta,
                     self.target_network.action: pad(action, self.target_network.action_size)}

        return sess.run([self.target_network.optimizer, self.target_network.loss], feed_dict)

    # def freeze_source(self):
    #     #     # Freeze all but U2 in source network
    #     #     self.source_networks[0].optimizer = tf.train.AdamOptimizer(learning_rate=self.source_networks[0].learning_rate) \
    #     #         .minimize(self.source_networks[0].loss, var_list=['U1'])
    #     #
    #     #     self.source_networks[1].optimizer = tf.train.AdamOptimizer(learning_rate=self.source_networks[0].learning_rate) \
    #     #         .minimize(self.source_networks[1].loss, var_list=['U1'])
    #
    #     # def unfreeze_source(self):
    #     #     # Unfreeze all in source network
    #     #     self.source_network.optimizer = tf.train.AdamOptimizer(learning_rate=self.source_network.learning_rate) \
    #     #         .minimize(self.source_network.loss)

    def init_output_weights(self):
        tf.variables_initializer(['W2', 'b2'])

    def set_env(self, env_name):
        self.curr_env_name = env_name

    def process(self, state):
        if self.curr_env_name == ENVS['mcc']:
            return self.scaler.transform(state)
            # return np.array([(state[0, 0]+1.2)/1.8, (state[0, 1]+0.07)/0.14]).reshape(1, -1)
        return state
