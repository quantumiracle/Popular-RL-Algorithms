"""
Proximal Policy Optimization for discrete (action space) environments, without GAE.
"""
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

#Hyperparameters
learning_rate = 0.0005
gamma         = 0.98
lmbda         = 0.95
eps_clip      = 0.1
K_epoch       = 3
T_horizon     = 20

class PPO(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PPO, self).__init__()
        self.data = []
        
        self.fc1   = nn.Linear(state_dim,256)
        self.fc_pi = nn.Linear(256,action_dim)
        self.fc_v  = nn.Linear(256,1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.mseLoss = nn.MSELoss()

    def pi(self, x, softmax_dim = 0):
        x = F.relu(self.fc1(x))
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob
    
    def v(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v
      
    def put_data(self, transition):
        self.data.append(transition)
        
    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, prob_a, done = transition
            
            s_lst.append(s)
            a_lst.append(a)
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append(prob_a)
            done_lst.append([int(done)])
            
        s,a,r,s_prime,done, prob_a = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
                                          torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
                                          torch.tensor(done_lst, dtype=torch.float), torch.tensor(prob_a_lst)
        self.data = []
        return s, a, r, s_prime, done, prob_a
        
    def train_net(self):
        s, a, r, s_prime, done, prob_a = self.make_batch()

        rewards = []
        discounted_r = 0
        for reward, d in zip(reversed(r), reversed(done)):
            if d:
                discounted_r = 0
            discounted_r = reward + gamma * discounted_r
            rewards.insert(0, discounted_r)  # insert in front, cannot use append

        rewards = torch.tensor(rewards, dtype=torch.float32)
        if rewards.shape[0]>1:  # a batch with size 1 will cause 0 std 
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        rewards = rewards

        for _ in range(K_epoch):
            vs = self.v(s)
            advantage = rewards - vs.squeeze(dim=-1).detach()
            vs_target = rewards

            pi = self.pi(s, softmax_dim=-1)
            dist = Categorical(pi)
            dist_entropy = dist.entropy()
            log_p = dist.log_prob(a)
            ratio = torch.exp(log_p - torch.log(prob_a))  # a/b == exp(log(a)-log(b))

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * advantage
            # loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.v(s) , vs_target.detach()) - 0.01*dist_entropy 
            loss = -torch.min(surr1, surr2) + 0.5*self.mseLoss(vs.squeeze(dim=-1) , vs_target.detach()) - 0.01*dist_entropy

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
        
def main():
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n  # discrete
    model = PPO(state_dim, action_dim)
    score = 0.0
    epi_len = []
    print_interval = 20

    for n_epi in range(10000):
        s = env.reset()
        done = False
        while not done:
            for t in range(T_horizon):
                prob = model.pi(torch.from_numpy(s).float())
                m = Categorical(prob)
                a = m.sample().item()
                s_prime, r, done, info = env.step(a)
                # env.render()
                model.put_data((s, a, r/100.0, s_prime, prob[a].item(), done))

                s = s_prime

                score += r
                if done:
                    break

            model.train_net()
        epi_len.append(t)
        if n_epi%print_interval==0 and n_epi!=0:
            print("# of episode :{}, avg score : {:.3f}, avg epi length :{}".format(n_epi, score/print_interval, int(np.mean(epi_len))))
            score = 0.0
            epi_len = []

    env.close()

if __name__ == '__main__':
    main()