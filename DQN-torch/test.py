import gymnasium as gym
import time
import numpy as np
import argparse
import matplotlib.pyplot as plt
from dqn_agent import DQNAgent 
import os
from collections import deque

gym.logger.set_level(40)

ACTION_SPACE = [
    (0, 0, 0), (0.6, 0, 0), (-0.6, 0, 0), (0, 0.2, 0), (0, 0, 0.8),  # (Steering Wheel, Gas, Brake)
] # do nothing, left, right, gas, brake

class Env:
    def __init__(self, action_stack=1, render=False):
        if render:
            self.env = gym.make('CarRacing-v2', render_mode='human')
        else:
            self.env = gym.make('CarRacing-v2')
        self.action_stack = action_stack

    def reset(self):
        state, _ = self.env.reset()
        for _ in range(30):
            state, _, _, _, _ = self.env.step(np.array([0, 0, 0]))
        self.reward_list = [0] * 100
        state = state[:84, 6:90]
        return np.moveaxis(state, -1, 0) / 255.0

    def step(self, action):
        total_reward = 0
        for _ in range(self.action_stack):
            state, reward, done, done_, _ = self.env.step(action)
            total_reward += reward
            self.update_reward(reward)
            if done or done_ or np.mean(self.reward_list) <= -0.1:
                done = True
                break
        state = state[:84, 6:90]
        return np.moveaxis(state, -1, 0) / 255.0, total_reward, done

    def update_reward(self, r):
        self.reward_list.pop(0)
        self.reward_list.append(r)
        assert len(self.reward_list) == 100

def dqn_train(env, agent, n_episode=1000, batch_size=64, early_stop_threshold=900):
    scores = []
    losses = []
    epsilons = []
    total_steps = 0
    best_score = float("-inf")

    for episode in range(n_episode):
        episode_steps = 0
        total_reward = 0
        total_loss = 0  # Accumulate loss
        loss_count = 0  # Count the number of training steps with valid loss

        state = env.reset()

        while True:
            action_index = agent.act(state)
            action = ACTION_SPACE[action_index]
            next_state, reward, done = env.step(action)

            total_steps += 1
            episode_steps += 1
            total_reward += reward

            terminated = done
            truncated = not done and episode_steps >= env.env._max_episode_steps
            agent.memorize(state, action_index, reward, next_state, terminated, truncated)
            loss = agent.train_model(batch_size)

            if loss is not None:
                total_loss += loss
                loss_count += 1

            if done:
                break

            state = next_state

        avg_loss = total_loss / loss_count if loss_count > 0 else 0
        scores.append(total_reward)
        losses.append(avg_loss)
        epsilons.append(agent.epsilon)
        avg_score = np.mean(scores[-20:])  

        # Save model if new high average score, episode reward >= 600, or every 10 episodes
        if avg_score > best_score or total_reward >= 600 or episode % 10 == 0:
            agent.save_model(episode, avg_score, total_reward)
            best_score = avg_score if avg_score > best_score else best_score
            
        agent.decay_epsilon()
            
        print(f"Episode: {episode:04}, steps taken: {episode_steps:04}, total steps: {total_steps:07},",
              f"episode reward: {total_reward:1f}, avg reward: {avg_score:1f}, avg loss: {avg_loss:.6f}")
        
        print(f"Epsilon after episode {episode:04}: {agent.epsilon:.6f}")

        # Check early stopping condition
        if avg_score >= early_stop_threshold:
            print(f"Early stopping at episode {episode:04}, avg reward: {avg_score:.2f}")
            break

    return scores, losses, epsilons

def save_plots(scores, losses, epsilons, output_dir="plots"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    episodes = np.arange(len(scores))

    plt.figure()
    plt.plot(episodes, scores, label='Rewards')
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.legend()
    plt.savefig(os.path.join(output_dir, "rewards_plot.png"))
    plt.close()
    
    plt.figure()
    plt.plot(episodes, losses, label='Loss')
    plt.xlabel('Episodes')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(output_dir, "losses_plot.png"))
    plt.close()
    
    plt.figure()
    plt.plot(episodes, epsilons, label='Epsilon')
    plt.xlabel('Episodes')
    plt.ylabel('Epsilon')
    plt.legend()
    plt.savefig(os.path.join(output_dir, "epsilons_plot.png"))
    plt.close()

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
            action_index = agent.act(state)
            action = ACTION_SPACE[action_index]
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
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-t', '--test', help='testing model', action='store_true', required=False)
    args = parser.parse_args()

    train = True
    if args.test:
        train = False

    if train:
        print("... start training ...")
        env = Env()
        state_size = (3, 84, 84)
        action_size = len(ACTION_SPACE)
        agent = DQNAgent(state_size=state_size, action_size=action_size)
        scores, losses, epsilons = dqn_train(env, agent, n_episode=1000, early_stop_threshold=600)
        save_plots(scores, losses, epsilons)
    else:
        print("... start testing ...")
        env = Env(render=True)
        state_size = (3, 84, 84)
        action_size = len(ACTION_SPACE)
        agent = DQNAgent(state_size=state_size, action_size=action_size, epsilon=0)
        agent.load_model()
        scores = dqn_test(env, agent, n_episode=100)
        print(f"Mean score: {np.mean(scores)}, Std score: {np.std(scores)}")
        np.save("dqn_car_racing_scores_100", scores)
