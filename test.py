import gymnasium as gym
import numpy as np

env = gym.make('CarRacing-v2', render_mode='human', domain_randomize=False)

state, _ = env.reset()
state = state[:84, 6:90, :]
state = np.dot(np.array(state), [0.2989, 0.5870, 0.1140])

def is_out_of_track(img_rgb):
        # Check if the car is out of the track
        # Define the off-track color range (e.g., green areas)
        out_sum = (img_rgb[75, 35:48, 1][:2] > 200).sum() + (img_rgb[75, 35:48, 1][-2:] > 200).sum()
        return out_sum == 4

action = np.array([0,1,0])
for _ in range(1000):
    next_state, reward, terminated, truncated, _ = env.step(action)
    next_state = next_state[:84, 6:90, :]
    if is_out_of_track(next_state):
        print("out")
    else:
        print("not out")
    
