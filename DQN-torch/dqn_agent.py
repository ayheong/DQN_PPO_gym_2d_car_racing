import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
from replay_buffer import ReplayBuffer

class DQNAgent:
    def __init__(self, state_size, action_size, buffer_size=5000, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.99, learning_rate=0.001):
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
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        
    def memorize(self, state, action, reward, next_state, terminated, truncated):
        self.memory.add(state, action, reward, next_state, terminated, truncated)
        
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            # Generate a random action with the first value between -1 and 1, and the other two between 0 and 1
            action = np.array([np.random.uniform(-1.0, 1.0), np.random.uniform(0.0, 1.0), np.random.uniform(0.0, 1.0)])
        else:
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            self.model.eval()
            with torch.no_grad():
                q_values = self.model(state)
            self.model.train()
            action = q_values.cpu().numpy()[0]
        return action
    
    def build_model(self):
        class QNetwork(nn.Module):
            def __init__(self, input_shape, num_actions):
                super(QNetwork, self).__init__()
                self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=3, stride=2)
                self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
                self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=2)
                self.fc_input_dim = self._get_conv_output(input_shape)
                self.fc1 = nn.Linear(self.fc_input_dim, 512)
                self.fc2 = nn.Linear(512, 256)
                self.fc3 = nn.Linear(256, num_actions)

            def _get_conv_output(self, shape):
                o = torch.zeros(1, *shape)
                o = self.conv1(o)
                o = self.conv2(o)
                o = self.conv3(o)
                return int(np.prod(o.size()))

            def forward(self, x):
                x = torch.relu(self.conv1(x))
                x = torch.relu(self.conv2(x))
                x = torch.relu(self.conv3(x))
                x = x.view(x.size(0), -1)
                x = torch.relu(self.fc1(x))
                x = torch.relu(self.fc2(x))
                x = torch.tanh(self.fc3(x))
                return x
        
        return QNetwork((3, 84, 84), self.action_size)
    
    def train_model(self, batch_size):
        if len(self.memory) < batch_size:
            return

        states, actions, rewards, next_states, terminateds, truncateds = self.memory.sample(batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        terminateds = torch.FloatTensor(terminateds).to(self.device)
        truncateds = torch.FloatTensor(truncateds).to(self.device)

        q_values = self.model(states)
        next_q_values = self.model(next_states)
        
        q_values_target = q_values.clone().detach()
        
        for i in range(batch_size):
            if terminateds[i] or truncateds[i]:
                q_values_target[i][actions[i]] = rewards[i]
            else:
                q_values_target[i][actions[i]] = rewards[i] + self.gamma * torch.max(next_q_values[i])
        
        loss = self.criterion(q_values, q_values_target)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            print(f"Updated epsilon: {self.epsilon}")

    def save_model(self, filename='dqn_model'):
        directory = 'saves'
        if not os.path.exists(directory):
            os.makedirs(directory)
        filepath = os.path.join(directory, f"{filename}.pth")
        torch.save(self.model.state_dict(), filepath)
        print(f"Model saved to {filepath}")
