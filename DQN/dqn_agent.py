import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
from replay_buffer import ReplayBuffer



class DQNAgent:
    def __init__(self, state_size, action_size, buffer_size=8000, gamma=0.99, epsilon=1, epsilon_min=0.01, epsilon_decay=0.99, learning_rate=0.001, target_update=100):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = ReplayBuffer(max_size=buffer_size)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.build_model().to(self.device)
        self.target_model = self.build_model().to(self.device)
        self.update_target_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.SmoothL1Loss()  # Huber loss
        self.target_update = target_update
        self.step_counter = 0

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def memorize(self, state, action_index, reward, next_state, terminated, truncated):
        self.memory.add(state, action_index, reward, next_state, terminated, truncated)

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            action_index = np.random.randint(5)
        else:
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            self.model.eval()
            with torch.no_grad():
                q_values = self.model(state)
            self.model.train()
            action_index = torch.argmax(q_values).item()
        return action_index

    def build_model(self):
        class QNetwork(nn.Module):
            def __init__(self, input_shape, num_actions):
                super(QNetwork, self).__init__()
                self.conv1 = nn.Conv2d(input_shape[0], 16, kernel_size=3, stride=2)
                self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2)
                self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=2)
                self.fc_input_dim = self._get_conv_output(input_shape)
                self.fc1 = nn.Linear(self.fc_input_dim, 128)
                self.fc2 = nn.Linear(128, num_actions)
                self.apply(self.weights)
                
            def _get_conv_output(self, shape):
                o = torch.zeros(1, *shape)
                o = self.conv1(o)
                o = self.conv2(o)
                o = self.conv3(o)
                return int(np.prod(o.size()))
            @staticmethod
            def weights(m):
                if isinstance(m, nn.Conv2d):
                    nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                    nn.init.constant_(m.bias, 0.1)

            def forward(self, x):
                x = torch.relu(self.conv1(x))
                x = torch.relu(self.conv2(x))
                x = torch.relu(self.conv3(x))
                x = x.view(x.size(0), -1)
                x = torch.relu(self.fc1(x))
                output = self.fc2(x)
                return output

        return QNetwork((3, 84, 84), 5)

    def train_model(self, batch_size):
        if len(self.memory) < batch_size:
            return

        # Sample a batch from the replay buffer
        states, actions, rewards, next_states, terminateds, truncateds = self.memory.sample(batch_size)

        # Convert to tensors and move to the device
        states = torch.FloatTensor(states).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        terminateds = torch.FloatTensor(terminateds).to(self.device)

        # Compute current Q values
        q_values = self.model(states)
        current_q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute next Q values for the next states using the target model
        next_q_values = self.target_model(next_states).max(1)[0].detach()

        # Compute target Q values
        target_q_values = rewards + (self.gamma * next_q_values * (1 - terminateds))

        # Compute the loss
        loss = self.criterion(current_q_values, target_q_values)

        # Perform gradient descent
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target model periodically
        self.step_counter += 1
        if self.step_counter % self.target_update == 0:
            self.update_target_model()

        return loss.item()

    def save_model(self, episode, filename='dqn_model'):
        directory = 'saves'
        if not os.path.exists(directory):
            os.makedirs(directory)
        filepath = os.path.join(directory, f"{filename}.pth")
        torch.save(self.model.state_dict(), filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filename='dqn_model'):
        filepath = os.path.join('saves', f"{filename}.pth")
        self.model.load_state_dict(torch.load(filepath))
        self.target_model.load_state_dict(torch.load(filepath))
        print(f"Model loaded from {filepath}")
        
    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay