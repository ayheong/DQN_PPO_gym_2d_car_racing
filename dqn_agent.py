import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Input, Conv2d, MaxPooling2D, Flatten, Dense
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
    
    def build_model(self):
        """
        Build the neural network model for approximating Q-values.
        
        Returns:
            model (Sequential): The compiled Keras model.
        """
        model = Sequential()
        model.add(Input(shape=(96, 96, 3)))
        model.add(Conv2d(32, kernel_size=(3,3), input_dim=self.state_size, activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Conv2d(64, kernel_size=(3,3), input_dim=self.state_size, activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Conv2d(64, kernel_size=(3,3), input_dim=self.state_size, activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss="mse", optimizer=Adam(learning_rate=self.learning_rate))
        return model