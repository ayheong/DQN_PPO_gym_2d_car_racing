import gymnasium as gym
import argparse
import time
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.beta import Beta

gym.logger.set_level(40)


class WrapperEnv:
    def __init__(self, env, sample_f=10):
        self.env = gym.make(env, verbose=0)
        self.sample_f = sample_f

    def reset(self):
        img_rgb, _ = self.env.reset()
        self.flag = [0] * 100
        return np.moveaxis(img_rgb, -1, 0) / 255.0

    def step(self, action):
        total_reward = 0
        for i in range(self.sample_f):
            img_rgb, reward, done, done_, _ = self.env.step(action)

            total_reward += reward
            self.update_flag(reward)

            if done or done_ or np.mean(self.flag) <= -0.1:
                done = True
                break

        return np.moveaxis(img_rgb, -1, 0) / 255.0, total_reward, done, done

    def update_flag(self, r):
        self.flag.pop(0)
        self.flag.append(r)
        assert len(self.flag) == 100


class Model(nn.Module):

    def __init__(self, obs_dim, act_dim, save_dir="./ppo_model"):
        super(Model, self).__init__()
        self.cnn_base = nn.Sequential(  # input shape (3, 96, 96)
            nn.Conv2d(obs_dim, 8, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2),  # (8, 47, 47)
            nn.ReLU(),  # activation
            nn.Conv2d(16, 32, kernel_size=3, stride=2),  # (16, 23, 23)
            nn.ReLU(),  # activation
            nn.Conv2d(32, 64, kernel_size=3, stride=2),  # (32, 11, 11)
            nn.ReLU(),  # activation
            nn.Conv2d(64, 128, kernel_size=3, stride=1),  # (64, 5, 5)
            nn.ReLU(),  # activation
            nn.Conv2d(128, 256, kernel_size=3, stride=1),  # (128, 3, 3)
            nn.ReLU(),  # activation
            nn.Flatten(),
        )  # output shape (256, 1, 1)
        self.v = nn.Sequential(nn.Linear(256, 100), nn.ReLU(), nn.Linear(100, 1))
        self.fc = nn.Sequential(nn.Linear(256, 100), nn.ReLU())
        self.alpha_head = nn.Sequential(nn.Linear(100, act_dim), nn.Softplus())
        self.beta_head = nn.Sequential(nn.Linear(100, act_dim), nn.Softplus())
        self.apply(self._weights_init)

        self.ckpt_file = save_dir + ".pth"

    @staticmethod
    def _weights_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(m.bias, 0.1)

    def forward(self, x):
        x = self.cnn_base(x)
        v = self.v(x)
        x = self.fc(x)
        alpha = self.alpha_head(x) + 1
        beta = self.beta_head(x) + 1
        return (alpha, beta), v

    def save_ckpt(self):
        torch.save(self.state_dict(), self.ckpt_file)

    def load_ckpt(self):
        self.load_state_dict(torch.load(self.ckpt_file))


class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.probs = []
        self.rewards = []
        self.next_states = []
        self.values = []

    def store(self, state, action, prob, reward, next_state, value):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(prob)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.values.append(value)

    def clear(self):
        self.states = []
        self.actions = []
        self.probs = []
        self.rewards = []
        self.next_states = []
        self.values = []

    def generate_batch(self, batch_size):
        length = len(self.states)
        starts = np.arange(0, length, batch_size)
        indices = np.arange(length, dtype=np.int32)

        batchs = [indices[i:i+batch_size] for i in starts]

        return self.states, self.actions, self.probs, self.rewards, self.next_states, self.values, batchs


class Agent:
    def __init__(self, state_dim, action_dim, gamma=0.99, lamda=0.95, clip=0.1,
                 learning_rate=0.001, batch_size=128, save_dir='./ppo_model',
                 epochs=8, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):

        self.model = Model(state_dim, action_dim, save_dir).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        self.gamma = gamma
        self.lamda = lamda
        self.clip = clip

        self.batch_size = batch_size
        self.epochs = epochs

        self.buffer = Memory()

        self.device = device

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float).to(self.device).unsqueeze(0)

        with torch.no_grad():
            alpha, beta = self.model(state)[0]
            value = self.model(state)[1]

        dist = Beta(alpha, beta)

        action = dist.sample()
        logp = dist.log_prob(action).sum(dim=1)

        action = action.squeeze().cpu().numpy()
        logp = logp.item()

        return action, logp, value

    def store(self, tranjectory):
        self.buffer.store(*tranjectory)

    def save_model(self):
        print("... save model ...")
        self.model.save_ckpt()

    def load_model(self):
        print("... load model ...")
        self.model.load_ckpt()

    def learn(self):
        states, actions, probs, rewards, next_states, values, batchs = self.buffer.generate_batch(self.batch_size)

        s = torch.tensor(states, dtype=torch.float).to(self.device)
        a = torch.tensor(actions, dtype=torch.float).to(self.device)
        old_logp = torch.tensor(probs, dtype=torch.float).to(self.device)
        r = torch.tensor(rewards, dtype=torch.float).to(self.device)
        next_s = torch.tensor(next_states, dtype=torch.float).to(self.device)

        v = torch.tensor(values, dtype=torch.float).to(self.device)
        advantages = torch.zeros_like(v)

        for i in tqdm(range(len(values)-1), total=len(values)-1):
            discount = 1
            a_t = 0
            for j in range(i, len(values)-1):
                a_t += discount*(r[j] + self.gamma*v[j+1] - v[j])
                discount *= self.gamma*self.lamda

            advantages[i] = a_t

        v_ = advantages+v  # returns

        advantages = advantages.view(-1, 1)
        v_ = v_.view(-1, 1)

        # print(advantages)
        # print(v_)

        for _ in range(self.epochs):
            for index in batchs:
                alpha, beta = self.model(s[index])[0]
                dist = Beta(alpha, beta)
                new_logp = dist.log_prob(a[index]).sum(dim=1, keepdim=True)
                ratio = new_logp.exp() / old_logp.view(-1, 1)[index].exp()

                surr1 = ratio * advantages[index]
                surr2 = torch.clamp(ratio, 1. - self.clip, 1. + self.clip) * advantages[index]

                actor_loss = -torch.min(surr1, surr2).mean()  # maximize advantage

                critic_loss = ((self.model(s[index])[1] - v_[index]) ** 2).mean()  # minimize diff between critics

                total_loss = actor_loss + critic_loss

                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

        self.buffer.clear()


def ppo_train(env, agent, n_episode=1000, update_step=2000):
    scores = []
    total_steps = 0
    learn_steps = 0
    best_score = float("-inf")

    time_start = time.time()

    for episode in range(n_episode):
        episode_steps = 0
        total_reward = 0

        state = env.reset()

        while True:
            action, logp, value = agent.select_action(state)
            action_ = action * np.array([2., 1., 1.]) + np.array([-1., 0., 0.])
            next_state, reward, done, info = env.step(action_)

            total_steps += 1
            episode_steps += 1
            total_reward += reward
            agent.store((state, action, logp, reward, next_state, value))

            if total_steps % update_step == 0:
                print("...updating...")
                agent.learn()
                learn_steps += 1

            if done:
                break

            state = next_state

        scores.append(total_reward)
        avg_score = np.mean(scores[-100:])

        if avg_score > best_score:
            agent.save_model()
            best_score = avg_score

        s = int(time.time() - time_start)
        print(f"Ep. {episode:04}, Ep.s {episode_steps:04}, Total.s {total_steps:07}, L.s {learn_steps:04},",
              f"R. {total_reward:1f}, Avg. {avg_score:1f}, Time: {s // 3600:02}:{s % 3600 // 60:02}:{s % 60:02}")

    return scores


def ppo_test(env, agent, n_episode=500):
    scores = []
    total_steps = 0
    learn_steps = 0
    best_score = float("-inf")

    time_start = time.time()

    for episode in range(n_episode):
        episode_steps = 0
        total_reward = 0

        state = env.reset()

        while True:
            action, _, _ = agent.select_action(state)
            action_ = action * np.array([2., 1., 1.]) + np.array([-1., 0., 0.])
            next_state, reward, done, _ = env.step(action_)

            total_steps += 1
            episode_steps += 1
            total_reward += reward
            # agent.store((state, action, logp, reward, next_state, value))

            if done:
                break

            state = next_state

        scores.append(total_reward)
        avg_score = np.mean(scores[-100:])

        if avg_score > best_score:
            best_score = avg_score

        s = int(time.time() - time_start)
        print(f"Ep. {episode:04}, Ep.s {episode_steps:04}, Total.s {total_steps:07}, L.s {learn_steps:04},",
              f"R. {total_reward:1f}, Avg. {avg_score:1f}, Time: {s // 3600:02}:{s % 3600 // 60:02}:{s % 60:02}")

    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="arg parser")
    parser.add_argument('-m', '--mode', type=str, default="train", required=False, help="train/test?")
    parser.add_argument('-p', '--pretrain', action="store_true", help="pretrained model")
    args = parser.parse_args()

    if args.mode == "train":
        print("... start training ...")
        env = WrapperEnv('CarRacing-v2', sample_f=1)
        agent = Agent(state_dim=3, action_dim=3)
        if args.pretrain:
            agent.load_model()
        score = ppo_train(env, agent, 30000, 2000)
    elif args.mode == "test":
        print("... start testing ...")
        env = WrapperEnv('CarRacing-v2', 1)
        agent = Agent(state_dim=3, action_dim=3, save_dir='./ppo_model_demo')
        agent.load_model()
        scores = ppo_test(env, agent, n_episode=100)
        print(f"scores mean:{np.mean(scores)}, score std:{np.std(scores)}")
        np.save("ppo_car_racing_scores_100", scores)
    else:
        raise NotImplementedError