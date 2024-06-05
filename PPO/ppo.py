from env import Env
from buffer import Memory
import numpy as np
import argparse 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.beta import Beta
import os 
import matplotlib.pyplot as plt

class Model(nn.Module):
    def __init__(self, obs_dim, act_dim, save_dir="./ppo_model"):
        super(Model, self).__init__()
        self.cnn_base = nn.Sequential(  
            nn.Conv2d(obs_dim, 8, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2),  
            nn.ReLU(),  # activation
            nn.Conv2d(16, 32, kernel_size=3, stride=2),  
            nn.ReLU(),  # activation
            nn.Conv2d(32, 64, kernel_size=3, stride=2),  
            nn.ReLU(),  # activation
            nn.Conv2d(64, 128, kernel_size=3, stride=1), 
            nn.ReLU(),  # activation
            nn.Conv2d(128, 256, kernel_size=2, stride=1), 
            nn.ReLU(),  # activation
            nn.Flatten(), 
        )
        self.v = nn.Sequential(nn.Linear(256, 100), nn.ReLU(), nn.Linear(100, 1))
        self.fc = nn.Sequential(nn.Linear(256, 100), nn.ReLU())
        self.alpha = nn.Sequential(nn.Linear(100, act_dim), nn.Softplus())
        self.beta = nn.Sequential(nn.Linear(100, act_dim),nn.Softplus())

        self.apply(self.weights)
        self.ckpt_file = save_dir + ".pth"

    @staticmethod
    def weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(m.bias, 0.1)

    def forward(self, x):
        x = self.cnn_base(x)

        v = self.v(x)
        x = self.fc(x)

        return (self.alpha(x) + 1, self.beta(x) + 1), v

    def save_ckpt(self):
        
        torch.save(self.state_dict(), self.ckpt_file)

    def load_ckpt(self, device):
        self.load_state_dict(torch.load(self.ckpt_file, map_location=device))

class Agent:
    def __init__(self, state_dim, action_dim, gamma=0.99, lamda=0.95, clip=0.1,
                 learning_rate=1e-3, batch_size=128, save_dir='./ppo_model',
                 epochs=8, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):

        self.gamma = gamma # Discount factor
        self.lamda = lamda # GAE (Generalized Advantage Estimation) factor
        self.clip = clip # Clipping parameter
        self.batch_size = batch_size
        self.epochs = epochs
        self.buffer = Memory()
        self.device = device
        self.model = Model(state_dim, action_dim, save_dir).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float).to(self.device).unsqueeze(0)
        with torch.no_grad():
            alpha, beta = self.model(state)[0]
            value = self.model(state)[1]
        dist = Beta(alpha, beta)
        action = dist.sample()
        logp = dist.log_prob(action).sum(dim=1)  # Log probability of the action
        action = action.squeeze().cpu().numpy()
        logp = logp.item()
        return action, logp, value

    def memory(self, tranjectory):
        self.buffer.memory(*tranjectory)

    def save_model(self):
        print("... save model ...")
        self.model.save_ckpt()

    def load_model(self):
        print("... load model ...")
        self.model.load_ckpt(self.device)

    def learn(self):
        states, actions, probs, rewards, next_states, values, batchs = self.buffer.generate_batch(self.batch_size)
        
        s = torch.tensor(np.array(states), dtype=torch.float).to(self.device)
        a = torch.tensor(np.array(actions), dtype=torch.float).to(self.device)
        old_logp = torch.tensor(np.array(probs), dtype=torch.float).to(self.device)
        r = torch.tensor(np.array(rewards), dtype=torch.float).to(self.device)
        v = torch.tensor(values, dtype=torch.float).to(self.device)

        advantages = torch.zeros_like(v)
        # Compute advantages using Generalized Advantage Estimation (GAE)
        for i in range(len(values)-1):
            discount = 1
            a_t = 0
            for j in range(i, len(values)-1):
                a_t += discount*(r[j] + self.gamma*v[j+1] - v[j])
                discount *= self.gamma*self.lamda
            advantages[i] = a_t

        v_ = advantages + v  

        advantages = advantages.view(-1, 1)
        v_ = v_.view(-1, 1)

        for _ in range(self.epochs):
            for index in batchs:
                
                alpha, beta = self.model(s[index])[0]
                dist = Beta(alpha, beta)
                new_logp = dist.log_prob(a[index]).sum(dim=1, keepdim=True)
                ratio = (new_logp - old_logp.view(-1, 1)[index]).exp()
                
                # Surrogate objective
                surr1 = ratio * advantages[index]
                surr2 = torch.clamp(ratio, 1. - self.clip, 1. + self.clip) * advantages[index]

                p_loss = -torch.min(surr1, surr2).mean() # Policy loss
                v_loss = ((self.model(s[index])[1] - v_[index]) ** 2).mean()  # Value loss

                loss = p_loss + v_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        
        self.buffer.clear()
        return p_loss.item(), v_loss.item()


def ppo_train(env, agent, n_episode, update_step):
    scores = []
    score_list = []
    loss = []
    total_steps = 0
    learn_steps = 0
    best_score = float("-inf")

    for episode in range(n_episode):
        episode_steps = 0
        total_reward = 0

        state = env.reset()
        
        while True:
            action, logp, value = agent.select_action(state)
            act = action * np.array([2., 1., 1.]) + np.array([-1., 0., 0.]) # ensure the wheel is in between -1 to 1
            next_state, reward, done = env.step(act)

            total_steps += 1
            episode_steps += 1
            total_reward += reward
            agent.memory((state, action, logp, reward, next_state, value))

            if total_steps % update_step == 0:
                print("...updating...")
                p_loss, v_loss = agent.learn()
                learn_steps += 1
                loss.append((p_loss, v_loss))
            if done:
                break
            state = next_state

        scores.append(total_reward)
        avg_score = np.mean(scores[-100:]) # take the average of the last 10 elements
        
        score_list.append((total_reward, avg_score))
        if avg_score > best_score:
            agent.save_model()
            best_score = avg_score
        
    
        print(f"Epsode: {episode:04}, epsode steps: {episode_steps:04}, total steps: {total_steps:07}, learn steps: {learn_steps:04},",
              f"episode reward: {total_reward:1f}, avg reward: {avg_score:1f}")
        
    if avg_score == 850:
        return score_list, loss
   
    return score_list, loss

def ppo_test(env, agent, n_episode):
    scores = []
    total_steps = 0
    learn_steps = 0
    best_score = float("-inf")

    for episode in range(n_episode):
        episode_steps = 0
        total_reward = 0

        state = env.reset()

        while True:
            action, _, _ = agent.select_action(state)
            action[0] = action[0] * 2 - 1 # make sure the range of the wheel is from -1 to 1
            next_state, reward, done = env.step(action)
            total_steps += 1
            episode_steps += 1
            total_reward += reward
            if done:
                break
            state = next_state
        
        avg_score = np.mean(scores[-10:])
        scores.append(avg_score)
        if avg_score > best_score:
            best_score = avg_score

        print(f"Epsode: {episode:04}, epsode steps: {episode_steps:04}, total steps: {total_steps:07}, learn steps: {learn_steps:04},",
              f"episode reward: {total_reward:1f}, avg reward: {avg_score:1f}")
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
        agent = Agent(state_dim=3, action_dim=3)
        score, loss = ppo_train(env, agent, n_episode=1000, update_step=2000)

        e_score = [item[0] for item in score]
        a_score = [item[1] for item in score]
        plt.figure()
        plt.plot(e_score, 'b-', label='Episode Score')
        plt.plot(a_score, 'g-', label='Average Score')
        plt.xlabel('Episode')
        plt.ylabel('Score')
        plt.title('PPO Training Score')
        plt.grid(True)
        plt.legend()
        plt.savefig('ppo_training_score.png')


        p_loss = [item[0] for item in loss]
        v_loss = [item[1] for item in loss]
        
        tot_loss = p_loss + v_loss

        plt.figure()
        plt.plot(p_loss, 'r-', label='Policy Loss')
        plt.xlabel('Learning Steps')
        plt.ylabel('Loss')
        plt.title('PPO Training Loss')
        plt.grid(True)
        plt.legend()
        plt.savefig('ppo_policy_loss.png')

        plt.figure()
        plt.plot(v_loss, 'g-', label='Value Loss')
        plt.xlabel('Learning Steps')
        plt.ylabel('Loss')
        plt.title('PPO Training Loss')
        plt.grid(True)
        plt.legend()
        plt.savefig('ppo_value_loss.png')

        plt.figure()
        plt.plot(tot_loss, 'b-', label='Total Loss')
        plt.xlabel('Learning Steps')
        plt.ylabel('Loss')
        plt.title('PPO Training Loss')
        plt.grid(True)
        plt.legend()
        plt.savefig('ppo_total_loss.png')


    else:
        print("... start testing ...")
        env = Env(render=True)
        agent = Agent(state_dim=3, action_dim=3, save_dir='./ppo_model')
        agent.load_model()
        scores = ppo_test(env, agent, n_episode=10)
        print(f"scores mean:{np.mean(scores)}, score std:{np.std(scores)}")
        np.save("ppo_car_racing_scores_100", scores)
    