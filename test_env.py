import gymnasium as gym
import numpy as np
from replay_buffer import ReplayBuffer

env = gym.make('CarRacing-v2', render_mode='human')

print("Action space:", env.action_space)
print("Observation space:", env.observation_space)

state = env.reset()

for _ in range(100000):
    env.render()  

    action = env.action_space.sample()
    
    next_state, reward, done, truncated, info = env.step(action)
    
    print(f"Reward: {reward}, Done: {done}")
    
    if done:
        state = env.reset()

env.close()