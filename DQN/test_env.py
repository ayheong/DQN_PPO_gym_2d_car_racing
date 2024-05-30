import gymnasium as gym
import numpy as np
from DQN.dqn_agent import DQNAgent

env = gym.make('CarRacing-v2', render_mode='human')
state_size = (96, 96, 3)
action_size = env.action_space.shape[0]

agent = DQNAgent(state_size=state_size, action_size=action_size)

episodes = 10  
batch_size = 32
save_interval = 5
frames_per_action = 5

for episode in range(episodes):
    state, _ = env.reset()
    state = np.array(state)
    terminated = False
    truncated = False
    episode_memory = []
    total_reward = 0

    while not (terminated or truncated):
        action = agent.act(state)
        
        for _ in range(frames_per_action):
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = np.array(next_state)

            total_reward += reward

            episode_memory.append((state, action, reward, next_state, terminated, truncated))
            state = next_state
            
            if terminated or truncated:
                break

    for experience in episode_memory:
        state, action, reward, next_state, terminated, truncated = experience
        agent.memorize(state, action, reward, next_state, terminated, truncated)

    agent.train_model(batch_size)

    print(f"Episode {episode + 1}/{episodes} completed with total reward: {total_reward}")
    
    if (episode + 1) % save_interval == 0:
        agent.save_model(f'dqn_model_episode_{episode + 1}')
    
agent.save_model('dqn_model_final')