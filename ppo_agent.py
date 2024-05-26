# ppo_agent.py
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import os
from replay_buffer import ReplayBuffer

class PPOAgent:
    def __init__(self, state_size, action_size, lr=0.0001, gamma=0.99, clip_ratio=0.1, epochs=10, batch_size=64, buffer_size=10000):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.epochs = epochs
        self.batch_size = batch_size

        self.actor = self.build_actor()
        self.critic = self.build_critic()
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        self.memory = ReplayBuffer(max_size=buffer_size)

    def build_actor(self):
        model = tf.keras.Sequential([
            layers.Input(shape=self.state_size),
            layers.Conv2D(32, (8, 8), strides=(4, 4), activation='relu'),
            layers.Conv2D(64, (4, 4), strides=(2, 2), activation='relu'),
            layers.Conv2D(64, (3, 3), strides=(1, 1), activation='relu'),
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.Dense(self.action_size, activation='linear')
        ])
        return model

    def build_critic(self):
        model = tf.keras.Sequential([
            layers.Input(shape=self.state_size),
            layers.Conv2D(32, (8, 8), strides=(4, 4), activation='relu'),
            layers.Conv2D(64, (4, 4), strides=(2, 2), activation='relu'),
            layers.Conv2D(64, (3, 3), strides=(1, 1), activation='relu'),
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.Dense(1, activation='linear')
        ])
        return model

    def act(self, state, training: bool = True) -> np.ndarray:
        state = np.expand_dims(state, axis=0)
        action = self.actor(state)
        if training:
            noise = np.random.normal(0, 0.2, size=self.action_size)
            action = action + noise
        action = action[0].numpy()
        return action
        
     

    def memorize(self, state, action, reward, next_state, terminated, truncated):
        self.memory.add(state, action, reward, next_state, terminated, truncated)

    def compute_advantages(self, rewards, values, next_values, dones):
        advantages = []
        gae = 0
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * next_values[i] * (1 - dones[i]) - values[i]
            gae = delta + self.gamma * 0.95 * (1 - dones[i]) * gae
            advantages.insert(0, gae)
        return advantages

    def train_model(self):
        if len(self.memory) < self.batch_size:
            return

        states, actions, rewards, next_states, dones, truncated = self.memory.sample(self.batch_size)
        states = np.array(states)
        next_states = np.array(next_states)
        
        values = self.critic(states)
        next_values = self.critic(next_states)
        advantages = self.compute_advantages(rewards, values, next_values, dones)
        advantages = np.array(advantages)

        for _ in range(self.epochs):
            with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
                pred_actions = self.actor(states)
                
                pred_actions = tf.maximum(pred_actions, 0)
                
                pred_values = self.critic(states)
                ratios = tf.exp(tf.reduce_sum(pred_actions * actions, axis=1) - tf.reduce_sum(actions, axis=1))
                clipped_ratios = tf.clip_by_value(ratios, 1 - self.clip_ratio, 1 + self.clip_ratio)
                loss_actor = -tf.reduce_mean(tf.minimum(ratios * advantages, clipped_ratios * advantages))
                loss_critic = tf.reduce_mean((rewards + self.gamma * next_values - pred_values) ** 2)

            grads_actor = tape1.gradient(loss_actor, self.actor.trainable_variables)
            grads_critic = tape2.gradient(loss_critic, self.critic.trainable_variables)
            self.actor_optimizer.apply_gradients(zip(grads_actor, self.actor.trainable_variables))
            self.critic_optimizer.apply_gradients(zip(grads_critic, self.critic.trainable_variables))

    def save_model(self, filename='ppo_model'):
        directory = 'saves'
        if not os.path.exists(directory):
            os.makedirs(directory)
        filepath = os.path.join(directory, f"{filename}")
        print(f"Model saved to {filepath}")
        self.actor.save(f'{filepath}_actor.h5')
        self.critic.save(f'{filepath}_critic.h5')

    def load_model(self, filename='ppo_model_final'):
        directory = 'saves'
        filepath = os.path.join(directory, f"{filename}")
        self.actor = tf.keras.models.load_model(f'{filepath}_actor.h5')
        self.critic = tf.keras.models.load_model(f'{filepath}_critic.h5')

    def is_out_of_track(self, state):
        out_sum = (state[75, 35:48, 1][:2] > 200).sum() + (state[75, 35:48, 1][-2:] > 200).sum()
        return out_sum == 4
