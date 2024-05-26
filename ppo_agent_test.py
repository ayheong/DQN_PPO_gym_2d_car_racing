import numpy as np
import tensorflow as tf
import os

from tensorflow import keras
from keras import Model
from keras.layers import Input, Conv2D, Flatten, Dense
from keras.optimizers import Adam
import gymnasium as gym
from replay_buffer import ReplayBuffer



class PPOAgent:
    def __init__(self, state_size, action_size, buffer_size=10000, gamma=0.99, epsilon=0.2, learning_rate=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = ReplayBuffer(max_size=buffer_size)
        self.gamma = gamma
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.policy_model, self.value_model = self.build_model()
        
    def memorize(self, state, action, reward, next_state, terminated, truncated):
        self.memory.add(state, action, reward, next_state, terminated, truncated)

    def act(self, state):
        if np.random.rand() <= self.epsilon:  # Explore
            wheel = np.random.normal(0, 0.5)
            wheel = np.clip(wheel, -1, 1)
            gas = np.random.beta(2, 1)
            break_ = np.random.beta(1, 2)
            return np.array([wheel, gas, break_])  # Random choice within action bounds
        
        state = np.expand_dims(state, axis=0)
        action_mean = self.policy_model.predict(state, verbose=0)[0]  # Exploit, verbose=0 to stop printing log 
        
        action = action_mean + np.random.normal(0, 1, size=self.action_size)
        return np.clip(action, -1.0, 1.0)

    def build_model(self):
        input_layer = Input(shape=self.state_size)
        
        x = Conv2D(32, (8, 8), strides=(4, 4), activation='relu')(input_layer)
        x = Conv2D(64, (4, 4), strides=(2, 2), activation='relu')(x)
        x = Conv2D(64, (3, 3), strides=(1, 1), activation='relu')(x)
        x = Flatten()(x)
        x = Dense(512, activation='relu')(x)

        policy_output = Dense(self.action_size, activation='softmax', name='policy_output')(x)
        policy_model = Model(inputs=input_layer, outputs=policy_output)
        
        value_output = Dense(1, name='value_output')(x)
        value_model = Model(inputs=input_layer, outputs=value_output)
        
        policy_optimizer = Adam(learning_rate=self.learning_rate)
        value_optimizer = Adam(learning_rate=self.learning_rate)
    
    
        policy_model.compile(optimizer=policy_optimizer, loss=self.ppo_loss)
        value_model.compile(optimizer=value_optimizer, loss='mse')
        
        return policy_model, value_model
    
    def ppo_loss(self, y_true, y_pred):
        print()
        y_true = tf.reshape(y_true, (-1, 1))
        y_pred = tf.reshape(y_pred, (-1,))
        
        advantage = tf.expand_dims(y_true[0], axis=-1)
        old_policy = tf.expand_dims(y_true[1:], axis=-1)

        action_prob = y_pred
        
        ratio = action_prob / (old_policy + 1e-10)
        clipped_ratio = tf.clip_by_value(ratio, 1 - self.epsilon, 1 + self.epsilon)
        surrogate_loss = tf.minimum(ratio * advantage, clipped_ratio * advantage)
        return -tf.reduce_mean(surrogate_loss)
    
   
    def train_model(self, batch_size):
        # Training code
        states, actions, rewards, next_states, dones, old_probs = self.memory.sample(batch_size)
        advantages = self.compute_advantages(states, rewards, next_states, dones)
        for _ in range(batch_size):
            self.policy_model.train_on_batch(states, [advantages, old_probs])
        value_targets = rewards + self.gamma * (1 - dones) * self.value_model.predict(next_states)
        self.value_model.train_on_batch(states, value_targets)
    
    def compute_advantages(self, states, rewards, next_states, dones):
        values = self.value_model.predict(states)
        next_values = self.value_model.predict(next_states)
        
        advantages = np.zeros_like(rewards, dtype=np.float32)
        last_advantage = 0.0
        
        for t in reversed(range(len(rewards))):
            td_error = rewards[t] + self.gamma * (1 - dones[t]) * next_values[t] - values[t]
            advantages[t] = last_advantage = td_error + self.gamma * 0.95 * (1 - dones[t]) * last_advantage
        
        return advantages
    
    def is_out_of_track(self, img_rgb):
        # Check if the car is out of the track
        # Define the off-track color range (e.g., green areas)
        out_sum = (img_rgb[75, 35:48, 1][:2] > 200).sum() + (img_rgb[75, 35:48, 1][-2:] > 200).sum()
        return out_sum == 4

    def save_model(self, filename='ppo_model.kearas'):
        directory = 'saves'
        if not os.path.exists(directory):
            os.makedirs(directory)
        filepath = os.path.join(directory, f"{filename}.h5")
        self.policy_model.save(filepath)
        print(f"Model saved to {filepath}")
