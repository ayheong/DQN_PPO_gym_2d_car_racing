import gymnasium as gym
import numpy as np

gym.logger.set_level(10)

class Env:
    def __init__(self, env, action_stack):
        self.env = gym.make(env, render_mode = 'human')
        self.action_stack = action_stack

    def reset(self):
        state, _ = self.env.reset()
        for _ in range (30):
            state, _, _, _ , _ = self.env.step(np.array([0, 0 ,0]))
        self.flag = [0] * 100
        state = state[:84, 6:90]
        return np.moveaxis(state, -1, 0) / 255.0
    
    def step(self, action):
        total_reward = 0
        for _ in range(self.action_stack):
            state, reward, terminated, truncated, _ = self.env.step(action)

            total_reward += reward
            self.update_flag(reward)

            if terminated or truncated or np.mean(self.flag) <= -0.1:
                done = True
                break
        state = state[:84, 6:90]
        return np.moveaxis(state, -1, 0) / 255.0, total_reward, done

    def update_flag(self, r):
        self.flag.pop(0)
        self.flag.append(r)
        assert len(self.flag) == 100