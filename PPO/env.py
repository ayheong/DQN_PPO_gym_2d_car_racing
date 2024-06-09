import gymnasium as gym
import numpy as np

class Env:
    def __init__(self, action_repeat = 4, render = False):
        if render: 
            self.env = gym.make('CarRacing-v2', render_mode = 'human')
        else:
            self.env = gym.make('CarRacing-v2')
        self.action_repeat = action_repeat

    def reset(self):
        state, _ = self.env.reset()
        for _ in range (30):
            state, _, _, _ , _ = self.env.step(np.array([0, 0 ,0]))
        self.reward_list = [0] * 100

        state = np.dot(state[..., :], [0.299, 0.587, 0.114]) 
        state = state / 128. - 1.
        state = state[:84, 6:90]
        self.stack = [state] * 4
        return np.array(self.stack)
    
    def step(self, action):
        total_reward = 0
        done = False
        for _ in range(self.action_repeat):
            state, reward, terminated, truncated, _ = self.env.step(action)
            if np.mean(state[:, :, 1]) > 185.0:
                reward -= 0.05
            total_reward += reward
            self.update_reward(reward)
            if terminated or truncated or np.mean(self.reward_list) <= -0.1:
                done = True
                break

        state = state[:84, 6:90]
        state = np.dot(state[..., :], [0.299, 0.587, 0.114]) / 128. -1.

        self.stack.pop(0)
        self.stack.append(state)
        assert len(self.stack) == 4

        return np.array(self.stack) , total_reward, done

    def update_reward(self, r):
        self.reward_list.pop(0)
        self.reward_list.append(r)
        assert len(self.reward_list) == 100