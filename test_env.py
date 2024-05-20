import gymnasium as gym
import numpy as np
from dqn_agent import DQNAgent

env = gym.make('CarRacing-v2', render_mode='human')
state_size = (96, 96, 3)
action_size = env.action_space.shape[0]

agent = DQNAgent(state_size=state_size, action_size=action_size)

episodes = 1000  
batch_size = 32

for episode in range(episodes):
    state, _ = env.reset()
    state = np.array(state)
    terminated = False
    truncated = False
    episode_memory = []
    total_reward = 0

    while not (terminated or truncated):
        action = agent.act(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        next_state = np.array(next_state)

        total_reward += reward

        episode_memory.append((state, action, reward, next_state, terminated, truncated))
        state = next_state

    for experience in episode_memory:
        state, action, reward, next_state, terminated, truncated = experience
        agent.memorize(state, action, reward, next_state, terminated, truncated)

    agent.train_model(batch_size)

    print(f"Episode {episode + 1}/{episodes} completed with total reward: {total_reward}")