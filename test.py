import gymnasium as gym
import numpy as np


env = gym.make('CarRacing-v2', render_mode = 'huamn')

state, _ = env.reset()
state = state[:84, 6:90, :]
state = np.dot(np.array(state), [0.2989, 0.5870, 0.1140])


action = np.array([0,0.2,0])

for _ in range(30):
    img_rgb, _, done, _, _ = env.step(np.array([0, 0, 0]))
     
for i in range(1000):
    next_state, reward, terminated, truncated, _ = env.step(action)
    next_state = next_state[:84, 6:90, :]
    
    
    
    
