import gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# Hyperparameters
EPISODES = 1000
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
GAMMA = 0.99
LR = 0.001
BATCH_SIZE = 64
MEMORY_SIZE = 100000
TARGET_UPDATE = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Neural Network for DQN
class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(4096, 512)
        self.fc2 = nn.Linear(512, num_actions)
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# Experience Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.stack(state), action, reward, np.stack(next_state), done
    
    def __len__(self):
        return len(self.buffer)
    

    
# Epsilon-greedy policy
def select_action(state, policy_net, epsilon, num_actions):
    if random.random() > epsilon:
        with torch.no_grad():
            return policy_net(torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)).argmax().item()
    else:
        return random.randrange(num_actions)

# Update the target network
def update_target(policy_net, target_net):
    target_net.load_state_dict(policy_net.state_dict())

# Training loop
def train_dqn(env, policy_net, target_net, optimizer, replay_buffer):
    epsilon = EPSILON_START
    for episode in range(EPISODES):
        state, _ = env.reset()
        state = np.transpose(state, (2, 0, 1))
        done = False
        total_reward = 0
        while not done:
            action = select_action(state, policy_net, epsilon, 5)
            next_state, reward, done, _, _ = env.step(action)
            next_state = np.transpose(next_state, (2, 0, 1))
            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            
            if len(replay_buffer) > BATCH_SIZE:
                batch_state, batch_action, batch_reward, batch_next_state, batch_done = replay_buffer.sample(BATCH_SIZE)
                batch_state = torch.tensor(batch_state, dtype=torch.float32, device=device)
                batch_action = torch.tensor(batch_action, device=device)
                batch_reward = torch.tensor(batch_reward, dtype=torch.float32, device=device)
                batch_next_state = torch.tensor(batch_next_state, dtype=torch.float32, device=device)
                batch_done = torch.tensor(batch_done, dtype=torch.float32, device=device)
                
                current_q_values = policy_net(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
                next_q_values = target_net(batch_next_state).max(1)[0]
                target_q_values = batch_reward + (GAMMA * next_q_values * (1 - batch_done))
                
                loss = nn.MSELoss()(current_q_values, target_q_values)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
        if episode % TARGET_UPDATE == 0:
            update_target(policy_net, target_net)
        print(f"Episode {episode} - Total Reward: {total_reward}")

# Evaluation
def evaluate_dqn(env, policy_net):
    state, _ = env.reset()
    state = np.transpose(state, (2, 0, 1))
    done = False
    total_reward = 0
    while not done:
        action = policy_net(torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)).argmax().item()
        next_state, reward, done, _ = env.step(action)
        state = np.transpose(next_state, (2, 0, 1))
        total_reward += reward
    return total_reward

if __name__ == "__main__":
    env = gym.make("CarRacing-v2", continuous = False)
    input_shape = (3, 96, 96)
    num_actions = 5  # Discrete actions: do nothing, left, right, gas, brake
    
    policy_net = DQN(input_shape, num_actions).to(device)
    target_net = DQN(input_shape, num_actions).to(device)
    update_target(policy_net, target_net)
    
    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    replay_buffer = ReplayBuffer(MEMORY_SIZE)
    
    train_dqn(env, policy_net, target_net, optimizer, replay_buffer)
    
    print("Evaluating trained model...")
    total_reward = evaluate_dqn(env, policy_net)
    print(f"Total Reward: {total_reward}")
