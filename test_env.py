import gymnasium as gym
import numpy as np
from dqn_agent import DQNAgent
from ppo_agent import PPOAgent
import tensorflow as tf
import cv2
from collections import deque

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    except RuntimeError as e:
        pass 
else:
    pass

PPO = True
DQN = False 
RENDER = True

env = None
agent = None
state_size = (84, 84, 1) # obs[:84, 6:90, :]

def preprocess_state(state):
    state = state[:84, 6:90, :]  # Crop
    state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)  # Convert to grayscale
    state = cv2.resize(state, (84, 84))  # Downsample
    state = state / 255.0  # Normalize
    return state

if RENDER:
    env = gym.make('CarRacing-v2', render_mode='human', domain_randomize=False)
else: 
    env = gym.make('CarRacing-v2', domain_randomize=False)
    
action_size = env.action_space.shape[0] 
 
if PPO: 
    agent = PPOAgent(state_size=state_size, action_size=action_size)
elif DQN: 
    agent = DQNAgent(state_size=state_size, action_size=action_size)

episodes = 500
batch_size = 64
save_interval = 5
frames_per_action = 4

for episode in range(episodes):
    state, _ = env.reset()
    for _ in range(30): # skip the first 30 frams for zoom in 
        state, _, _, _, _ = env.step(np.array([0, 0, 0]))
        
    state = preprocess_state(state)
    
    done = False 
    terminated = False
    truncated = False
    episode_memory = []
    total_reward = 0

    while not (done):
       
        
        action = agent.act(state)
        
        for _ in range(frames_per_action):
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            out_done = agent.is_out_of_track(next_state[:84, 6:90, :])
            
            total_reward += reward

            reward_fail = False
            
            if out_done:
                reward -= 10
            
            done = terminated or truncated or out_done or reward_fail
            
            terminated = done
            turncated = done
            
            next_state = preprocess_state(next_state)
            episode_memory.append((state, action, reward, next_state, terminated, truncated))
            state = next_state
            
            if done:
                break
        
    for experience in episode_memory:
        state, action, reward, next_state, terminated, truncated = experience
        agent.memorize(state, action, reward, next_state, terminated, truncated)
    agent.train_model()

    print(f"Episode {episode + 1}/{episodes} completed with total reward: {total_reward}")
    
    # if (episode + 1) % save_interval == 0:
    #     agent.save_model(f'ppo_model_episode_{episode + 1}')
    
agent.save_model('ppo_model_final')