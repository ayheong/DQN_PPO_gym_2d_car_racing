import numpy as np
class Memory:
    def __init__(self):
        self.states, self.actions, self.probs, self.rewards, self.next_states, self.values = [], [], [], [], [], []

    def store(self, state, action, prob, reward, next_state, value):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(prob)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.values.append(value)

    def clear(self):
        self.states, self.actions, self.probs, self.rewards, self.next_states, self.values = [], [], [], [], [], []

    def generate_batch(self, batch_size):
        length = len(self.states)
        starts = np.arange(0, length, batch_size)
        indices = np.arange(length, dtype=np.int32)

        batchs = [indices[i:i+batch_size] for i in starts]

        return self.states, self.actions, self.probs, self.rewards, self.next_states, self.values, batchs