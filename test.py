import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# Hyperparameters
LEARNING_RATE = 0.0003
GAMMA = 0.99
LAMBDA = 0.95
CLIP_EPS = 0.2
UPDATE_EPOCHS = 10
MINI_BATCH_SIZE = 128
TOTAL_TIMESTEPS = 1_000_000
ROLLOUT_SIZE = 2048

# Neural Network Architectures
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, output_dim)
    
    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return torch.softmax(self.fc3(x), dim=-1)

class ValueNetwork(nn.Module):
    def __init__(self, input_dim):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)
    
    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return self.fc3(x)

# PPO Update Function
def ppo_update(policy_net, value_net, optimizer_policy, optimizer_value, trajectories):
    states, actions, log_probs, returns, advantages = trajectories

    for _ in range(UPDATE_EPOCHS):
        for i in range(0, len(states), MINI_BATCH_SIZE):
            idx = slice(i, i + MINI_BATCH_SIZE)
            state_batch = states[idx]
            action_batch = actions[idx]
            old_log_probs_batch = log_probs[idx]
            return_batch = returns[idx]
            advantage_batch = advantages[idx]

            # Calculate current log probabilities and value estimates
            new_log_probs = Categorical(policy_net(state_batch)).log_prob(action_batch)
            value_estimates = value_net(state_batch).squeeze()

            # Calculate the ratio (pi_theta / pi_theta_old)
            ratios = torch.exp(new_log_probs - old_log_probs_batch)

            # Calculate surrogate losses
            surr1 = ratios * advantage_batch
            surr2 = torch.clamp(ratios, 1 - CLIP_EPS, 1 + CLIP_EPS) * advantage_batch

            # Actor loss
            actor_loss = -torch.min(surr1, surr2).mean()

            # Critic loss
            critic_loss = nn.MSELoss()(value_estimates, return_batch)

            # Total loss
            loss = actor_loss + 0.5 * critic_loss

            # Update policy network
            optimizer_policy.zero_grad()
            loss.backward()
            optimizer_policy.step()

            # Update value network
            optimizer_value.zero_grad()
            critic_loss.backward()
            optimizer_value.step()

# Collect trajectories
def collect_trajectories(env, policy_net, value_net):
    states, actions, rewards, log_probs = [], [], [], []
    state, _ = env.reset()
    done = False
    while not done:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        dist = Categorical(policy_net(state_tensor))
        action = dist.sample()
        log_prob = dist.log_prob(action)

        next_state, reward, done, _, _ = env.step(action.item())

        states.append(state)
        actions.append(action.item())
        rewards.append(reward)
        log_probs.append(log_prob.item())

        state = next_state

    return states, actions, rewards, log_probs

# Compute advantages and returns
def compute_advantages(rewards, values, next_values, dones):
    advantages = []
    returns = []
    gae = 0
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + GAMMA * next_values[step] * (1 - dones[step]) - values[step]
        gae = delta + GAMMA * LAMBDA * gae * (1 - dones[step])
        advantages.insert(0, gae)
        returns.insert(0, gae + values[step])
    return advantages, returns

# Training Loop
def train(env, policy_net, value_net, optimizer_policy, optimizer_value):
    state_dim = env.observation_space.shape[0]
    action_dim = 3

    for timestep in range(0, TOTAL_TIMESTEPS, ROLLOUT_SIZE):
        states, actions, rewards, log_probs = collect_trajectories(env, policy_net, value_net)

        # Convert lists to tensors
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        log_probs = torch.FloatTensor(log_probs)

        # Compute values and advantages
        values = value_net(states).detach().squeeze()
        next_values = torch.cat([values[1:], torch.zeros(1)])
        dones = torch.cat([torch.zeros(len(rewards)-1), torch.ones(1)])
        advantages, returns = compute_advantages(rewards, values, next_values, dones)

        # Update policy and value networks
        trajectories = (states, actions, log_probs, returns, advantages)
        ppo_update(policy_net, value_net, optimizer_policy, optimizer_value, trajectories)

# Initialize environment and networks
env = gym.make('CarRacing-v2')
state_dim = env.observation_space.shape[0]
action_dim = 3

policy_net = PolicyNetwork(state_dim, action_dim)
value_net = ValueNetwork(state_dim)

optimizer_policy = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
optimizer_value = optim.Adam(value_net.parameters(), lr=LEARNING_RATE)

# Train the agent
train(env, policy_net, value_net, optimizer_policy, optimizer_value)

# Save the trained model
torch.save(policy_net.state_dict(), 'ppo_car_racing_policy.pth')
torch.save(value_net.state_dict(), 'ppo_car_racing_value.pth')
