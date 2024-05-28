import numpy as np
import tensorflow as tf
import os
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from replay_buffer import ReplayBuffer

class DQNAgent():
    def __init__(self, state_size, action_size, buffer_size=2000, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, learning_rate=0.001):
        """
        Initialize the DQN agent with the given parameters.
        
        Args:
            state_size (int): The size of the state space (input layer).
            action_size (int): The size of the action space (output layer).
            buffer_size (int): The maximum size of the replay buffer.
            gamma (float): The discount factor for future rewards.
            epsilon (float): The initial exploration rate.
            epsilon_min (float): The minimum exploration rate.
            epsilon_decay (float): The decay rate for the exploration probability.
            learning_rate (float): The learning rate for the optimizer.
        """
        self.state_size = state_size
        self.action_size = action_size
        self.memory = ReplayBuffer(max_size=buffer_size)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.model = self.build_model()
        
    def memorize(self, state, action, reward, next_state, terminated, truncated):
        """
        Store a transition in the replay buffer.

        Args:
            See ReplayBuffer.add()
        """
        self.memory.add(state, action, reward, next_state, terminated, truncated)
        
    def act(self, state):
        """
        Select an action based on the current policy.
        
        Args:
            state (numpy array): The current state.
        
        Returns:
            numpy array: The action selected.
        """
        if np.random.rand() <= self.epsilon:  # Explore
            return np.random.uniform(-1.0, 1.0, self.action_size)  # Random Choice
        q_values = self.model.predict(state[np.newaxis, ...])  # Exploit
        return np.clip(q_values[0], -1.0, 1.0)  # 
    
    def build_model(self):
        """
        Build the neural network model for approximating Q-values.
        
        Returns:
            model (Sequential): The compiled Keras model.
        """
        model = Sequential()
        model.add(Input(shape=(96, 96, 3)))
        model.add(Conv2D(32, kernel_size=(3,3), input_dim=self.state_size, activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Conv2D(64, kernel_size=(3,3), input_dim=self.state_size, activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Conv2D(64, kernel_size=(3,3), input_dim=self.state_size, activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss="mse", optimizer=Adam(learning_rate=self.learning_rate))
        return model
    
    def train_model(self, batch_size):
        """
        Train the neural network using a batch of transitions from memory.
        
        Args:
            batch_size (int): The number of transitions to sample for training.
        """
        if len(self.memory) < batch_size:
            return

        states, actions, rewards, next_states, terminateds, truncateds = self.memory.sample(batch_size)
        
        states = np.array(states)
        next_states = np.array(next_states)
        
        q_values = self.model.predict(states)
        q_values_next = self.model.predict(next_states)
        
        for i in range(batch_size):
            if isinstance(actions[i], np.ndarray):
                action_index = np.argmax(actions[i]) 
            else:
                action_index = int(actions[i])
            if terminateds[i] or truncateds[i]:
                q_values[i][action_index] = rewards[i]
            else:
                q_values[i][action_index] = rewards[i] + self.gamma * np.amax(q_values_next[i])
        

        self.model.fit(states, q_values, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
    def save_model(self, filename='dqn_model'):
        """
        Save the neural network model to the specified file.
        
        Args:
            filename (str): The name of the file to save the model to.
        """
        directory = 'saves'
        if not os.path.exists(directory):
            os.makedirs(directory)
        filepath = os.path.join(directory, f"{filename}.h5")
        self.model.save(filepath)
        print(f"Model saved to {filepath}")