import numpy as np
from collections import deque

class ReplayBuffer: 
    def __init__(self, max_size):
        """
        Initialize the replay buffer with a maximum size.
        
        Args:
            max_size (int): Maximum number of actions to be stored in the buffer.
        """
        self.buffer = deque(maxlen=max_size)
        
    def __len__(self):
        """
        Returns the current number of transitions in the buffer.

        Returns:
            int: Current number of transitions in the buffer.
        """
        return len(self.buffer)
    
    def add(self, state, action, reward, next_state, terminated, truncated):
        """
        Adds a new transition to the buffer.

        Args:
            state (numpy array): The current state.
            action (int): The action taken.
            reward (float): The reward received.
            next_state (numpy array): The next state.
            terminated (bool): Whether the episode has ended due to a failure (ex. going off the track).
            truncated (bool): Whether the episode has ended due to a non-failure (ex. finishing a lap).
        """
        transition = (state, action, reward, next_state, terminated, truncated)
        self.buffer.append(transition)
        
    def sample(self, batch_size):
        """
        Randomly sample a batch of transitions from the buffer.
        
        Args:
            batch_size (int): The number of transition to sample.

        Returns:
            tuple: A tuple of numpy arrays (states, actions, rewards, next_states, terminated, truncated).
        """
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        sample = [self.buffer[i] for i in indices]
        return tuple(map(np.array, zip(*sample)))