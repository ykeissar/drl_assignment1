import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, InputLayer
import random
import numpy as np


class DQNAgent:
    C = 1000
    
    def __init__(self, env, n_layers=3, gamma=0.999, epsilon=0.5, update_target_model=5):
        self.env = env

        self.model = self.init_model(n_layers)

        self.q_val_weights = None
        self.target_weights = None
        self.curr_model_target = False

        self.experience_replay = []
        
        self.gamma = gamma
        self.epsilon = epsilon
        self.update_target_model = update_target_model

        self.terminal_state_counter = 0

    def init_model(self, n_layers):
        model = tf.keras.models.Sequential()
        model.add(InputLayer(input_shape=self.env.observation_space.shape, name='input'))

        for i in range(n_layers):
            model.add(Dense(64, activation='relu', name=f'dens{i}'))

        model.add(Dense(self.env.action_space.n,   activation='linear', name='output'))

        model.compile(loss="mse", optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
        return model

    def sample_batch(self, size=1000):
        random.shuffle(self.experience_replay)
        copy = self.experience_replay.copy()
        self.experience_replay = []
        return copy[:size]

    def sample_action(self, s):
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()

        actions = self.model.predict(s)
        return np.argmax(actions)

    def train_agent(self, terminal_state):
        if len(self.experience_replay) < DQNAgent.C:
            return

        minibatch = self.sample_batch()
        curr_states = list(replay[0] for replay in minibatch)

        self.switch_to_q_value()
        curr_qs = self.model.predict(np.array(curr_states))

        new_states = list(replay[3] for replay in minibatch)

        self.switch_to_target()
        new_qs = self.model.predict(np.array(new_states))

        X = []
        y = []
        for i, curr in enumerate(minibatch):
            curr_s, a, r, new_s, done = curr
            if done:
                target = r
            else:
                target = r + self.gamma * np.max(new_qs[i])

            new_curr_qs = curr_qs[i]
            new_curr_qs[a] = target

            X.append(curr_s)
            y.append(new_curr_qs)

        self.switch_to_q_value()
        self.model.fit(np.array(X), np.array(y))

        # TODO check this condition, if needed or not.
        if terminal_state or True:
            self.terminal_state_counter += 1

        if self.terminal_state_counter > self.update_target_model:
            self.terminal_state_counter = 0
            self.switch_to_q_value()
            self.target_weights = self.model.get_weights()

    def test_agent(self):
        s = self.env.reset()
        done = False
        total_rewards = 0
        while not done:
            a = self.sample_action(s)
            new_s, r, done, _ = self.env.step(a)
            total_rewards += r
            s = new_s
        pass

    def switch_to_q_value(self):
        '''
        Switch model's weights to q-value model IF currently in target.
        Backup target weights.
        :return:
        '''
        if self.curr_model_target:
            self.target_weights = self.model.get_weights()
            self.model.set_weights(self.q_val_weights)
            self.curr_model_target = False

    def switch_to_target(self):
        '''
        Switch model's weights to target model IF currently in q-value.
        Backup q-value weights.
        :return:
        '''
        if not self.curr_model_target:
            self.q_val_weights = self.model.get_weights()
            if not self.target_weights:
                self.target_weights = np.copy(self.q_val_weights)
            self.model.set_weights(self.target_weights)
            self.curr_model_target = True
