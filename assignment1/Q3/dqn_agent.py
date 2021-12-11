import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, InputLayer
import random
import numpy as np
from collections import deque
from tensorflow.keras.callbacks import TensorBoard
import time


class DQNAgent:
    def __init__ (self, env, n_layers=5, n_hidden_units=32, gamma=0.95, epsilon=1, epsilon_decay=0.99,
                 min_epsilon=0.001, lr=0.001, batch_size=48, update_target_model=20, tb=True):

        self.env = env
        self.n_states = env.observation_space.shape[0]
        self.n_actions = env.action_space.n

        self.q_val_weights = None
        self.target_weights = None
        self.curr_model_target = False

        self.experience_replay = deque([], maxlen=2500)
        self.experience_replay_memory = dict()
        self.experience_replay_memory_did_used = dict()
        self.memory_index = 0

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.lr = lr
        self.batch_size = batch_size

        self.model = self.init_model(n_layers, n_hidden_units)

        self.update_target_model = update_target_model
        self.training_counter = 0

        self.name = f"model-{n_layers}_L-{n_hidden_units}_U-{int(time.time())}"
        self.callbacks = [TensorBoard(log_dir=f"logs/{self.name}")] if tb else []

    def init_model(self, n_layers, n_hidden_units):
        model = tf.keras.models.Sequential()

        for i in range(n_layers):
            if i == 0:
                model.add(Dense(n_hidden_units, activation='relu', input_dim=self.n_states, name=f'dens{i}'))
            else:
                model.add(Dense(n_hidden_units, activation='relu', name=f'dens{i}'))

        model.add(Dense(self.n_actions, activation='linear', name='output'))

        model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=self.lr), metrics=['accuracy'])
        return model

    def sample_batch(self):
        experiences_to_use = list()
        keys_to_use = list()
        for key in self.experience_replay_memory_did_used.keys():
            if self.experience_replay_memory_did_used[key] == 0:
                keys_to_use.append(key)
        if len(experiences_to_use) >= self.batch_size:
            keys_chosen = random.sample(keys_to_use, self.batch_size)
            for exp_chosen in keys_chosen:
                experiences_to_use.append((self.experience_replay_memory[exp_chosen]))
                self.experience_replay_memory_did_used[key] = 1

        else:
            keys_rest = list()
            for key in self.experience_replay_memory_did_used.keys():
                if self.experience_replay_memory_did_used[key] == 1:
                    keys_rest.append(key)
            keys_chosen_addition = random.sample(keys_rest, self.batch_size - len(keys_to_use))
            for exp_chosen in keys_to_use:
                experiences_to_use.append((self.experience_replay_memory[exp_chosen]))
                self.experience_replay_memory_did_used[key] = 1
            for exp_chosen in keys_chosen_addition:
                experiences_to_use.append((self.experience_replay_memory[exp_chosen]))
                self.experience_replay_memory_did_used[key] = 1
        return experiences_to_use



    def sample_action(self, s):
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()

        actions = self.model.predict(s)
        return rand_argmax(actions)

    def train_agent(self):
        if len(self.experience_replay) < self.batch_size:
            return

        minibatch = self.sample_batch()
        curr_states = np.array(list(replay[0].reshape(4) for replay in minibatch))

        self.switch_to_q_value()
        curr_qs = self.model.predict(curr_states)

        new_states = np.array(list(replay[3].reshape(4) for replay in minibatch))

        self.switch_to_target()
        # new_qs = self.target_model.predict(np.array(new_states))
        new_qs = self.model.predict(new_states)

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
        np_X = np.array(X).reshape(self.batch_size, self.n_states)
        np_y = np.array(y)
        self.model.fit(np_X, np_y, verbose=0, callbacks=self.callbacks)

        self.training_counter += 1

        if self.training_counter > self.update_target_model:
            self.training_counter = 0
            self.switch_to_q_value()
            self.target_weights = self.model.get_weights()
            # self.target_model.set_weights(self.model.get_weights())

        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay

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
        """
        Switch model's weights to q-value model IF currently in target.
        Backup target weights.
        :return:
        """

        if self.curr_model_target:
            self.target_weights = self.model.get_weights()
            self.model.set_weights(self.q_val_weights)
            self.curr_model_target = False

    def switch_to_target(self):
        """
        Switch model's weights to target model IF currently in q-value.
        Backup q-value weights.
        :return:
        """

        if not self.curr_model_target:
            self.q_val_weights = self.model.get_weights()
            if not self.target_weights:
                self.target_weights = np.copy(self.q_val_weights)
            self.model.set_weights(self.target_weights)
            self.curr_model_target = True

    def add_to_replay(self, s, a, r, new_s, done):
        self.experience_replay.append((s, a, r, new_s, done))
        self.experience_replay_memory[self.memory_index] = (s, a, r, new_s, done)
        self.experience_replay_memory_did_used[self.memory_index] = 0
        self.memory_index = self.memory_index + 1


def rand_argmax(arr):
    return np.random.choice(np.flatnonzero(arr == arr.max()))