import gymnasium as gym
import time
import numpy as np

from dqn_agent import DQNAgent 

gym.logger.set_level(40)


class Env:
    def __init__(self, env, sample_f=4):
        self.env = gym.make(env, verbose=0, render_mode='human')
        self.sample_f = sample_f

    def reset(self):
        state, _ = self.env.reset()
        for _ in range(30):
            state, _, _, _, _ = self.env.step(np.array([0, 0, 0]))
        self.flag = [0] * 100
        state = state[:84, 6:90]
        return np.moveaxis(state, -1, 0) / 255.0
    
    def step(self, action):
        total_reward = 0
        for _ in range(self.sample_f):
            state, reward, done, done_, _ = self.env.step(action)
            
            total_reward += reward
            self.update_flag(reward)

            if done or done_ or np.mean(self.flag) <= -0.1:
                done = True
                break
        state = state[:84, 6:90]
        return np.moveaxis(state, -1, 0) / 255.0, total_reward, done

    def update_flag(self, r):
        self.flag.pop(0)
        self.flag.append(r)
        assert len(self.flag) == 100



def dqn_train(env, agent, n_episode=1000, batch_size=64):
    scores = []
    total_steps = 0
    best_score = float("-inf")

    for episode in range(n_episode):
        episode_steps = 0
        total_reward = 0

        state = env.reset()

        while True:
            action = agent.act(state)
            print(f"Action taken: {action}")  
            next_state, reward, done = env.step(action)
            
            # Gas and no brake
            if action[1] > 0 and action[2] == 0:
                reward *= 1.5

            total_steps += 1
            episode_steps += 1
            total_reward += reward

            terminated = done
            truncated = not done and episode_steps >= env.env._max_episode_steps
            agent.memorize(state, action, reward, next_state, terminated, truncated)
            agent.train_model(batch_size)

            if done:
                break

            state = next_state

        scores.append(total_reward)
        avg_score = np.mean(scores[-100:])

        if avg_score > best_score:
            agent.save_model()
            best_score = avg_score

        print(f"Episode: {episode:04}, steps taken: {episode_steps:04}, total steps: {total_steps:07},",
              f"episode reward: {total_reward:1f}, avg reward: {avg_score:1f}")

        agent.decay_epsilon()

    return scores


def dqn_test(env, agent, n_episode=500):
    scores = []
    total_steps = 0
    best_score = float("-inf")

    time_start = time.time()

    for episode in range(n_episode):
        episode_steps = 0
        total_reward = 0

        state = env.reset()

        while True:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            total_steps += 1
            episode_steps += 1
            total_reward += reward

            if done:
                break
            state = next_state

        scores.append(total_reward)
        avg_score = np.mean(scores[-100:])
        if avg_score > best_score:
            best_score = avg_score

        s = int(time.time() - time_start)
        print(f"Episode: {episode:04}, steps: {episode_steps:04}, total steps: {total_steps:07},",
              f"reward: {total_reward:1f}, avg reward: {avg_score:1f}, time: {s // 3600:02}:{s % 3600 // 60:02}:{s % 60:02}")
    return scores



if __name__ == "__main__":
    train = True
    if train:
        print("... start training ...")
        env = Env('CarRacing-v2', sample_f=4) 
        state_size = (3, 84, 84)
        action_size = 3
        agent = DQNAgent(state_size=state_size, action_size=action_size)
        scores = dqn_train(env, agent, n_episode=1000)

    else:
        print("... start testing ...")
        env = Env('CarRacing-v2', sample_f=4)  
        state_size = (3, 84, 84)
        action_size = 3
        agent = DQNAgent(state_size=state_size, action_size=action_size)
        agent.load_model()
        scores = dqn_test(env, agent, n_episode=100)
        print(f"Mean score: {np.mean(scores)}, Std score: {np.std(scores)}")
        np.save("dqn_car_racing_scores_100", scores)
