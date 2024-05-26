from ppo_agent import PPOAgent
import gymnasium as gym
import cv2
import numpy as np
from collections import deque
# Instantiate the agent
env = gym.make('CarRacing-v2', render_mode='human', domain_randomize=False)
state_size = state_size = (84, 84, 4)  # Define the state size of your environment
action_size = env.action_space.shape[0]   # Define the action size of your environment
agent = PPOAgent(state_size, action_size)

def preprocess_state(state):
    state = state[:84, 6:90, :]  # Crop
    state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)  # Convert to grayscale
    state = cv2.resize(state, (84, 84))  # Downsample
    state = state / 255.0  # Normalize
    return state

# Load the trained model
agent.load_model()
# Assuming you have an environment object named 'env'
state, _ = env.reset()
for _ in range(30): # skip the first 30 frams for zoom in 
    state, _, _, _, _ = env.step(np.array([0, 0, 0]))
state = preprocess_state(state)
state_stack = deque([state] * 4, maxlen=4)
while (True):
    # Generate action
    state_input = np.stack(state_stack, axis=-1)
    action = agent.act(state_input)
    
    # Execute action in the environment
    next_state, reward, done, _, _ = env.step(action)
    
    # Update state
    state = preprocess_state(next_state)
    state_stack.append(state)

# After the loop ends, the agent has completed its episode in the environment
