import tensorflow as tf
import numpy as np

class DQNAgent():
    def __init__(self, state_size, action_size, buffer_size=2000, gamma=0.95, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, learning_rate=0.001):
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