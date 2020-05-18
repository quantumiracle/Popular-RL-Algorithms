'''
Multi-processing for PPO continuous version 2

Several tricks need to be careful in multiprocess PPO:

* As PPO takes online training, the buffer contains sequential samples from rollouts,
so the buffer CANNOT be shared across processes, the sequece orders will be disturbed 
if the buffer is feeding with samples from different processes at the same time. Each process
can main its own buffer.

* A larger batch size usually ensures the stable training of PPO, also the update steps 
for both actor and critic need to be large if the training batch is large, because the agent
is learning from more samples in this case, which requires more training for each batch.

* Reward normalization can be critical. It could have significant effects for environments like
LunarLanderContinuous-v2, etc.

* The std of the action from the actor usually does no depend on the input state, which follows 
openai baseline implementation and other high-starred repository. 

* The optimization methods of 'kl_penal' and 'clip' are usually task-specific and empiracle

'''


import math
import random

import gym
import numpy as np

import torch
torch.multiprocessing.set_start_method('forkserver', force=True) # critical for make multiprocessing work
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal, MultivariateNormal

from IPython.display import clear_output
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import display
from reacher import Reacher

import argparse
import time

import torch.multiprocessing as mp
from torch.multiprocessing import Process

from multiprocessing import Process, Manager
from multiprocessing.managers import BaseManager

import threading as td

GPU = True
device_idx = 0
if GPU:
    device = torch.device("cuda:" + str(device_idx) if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
print(device)


parser = argparse.ArgumentParser(description='Train or test neural net motor controller.')
parser.add_argument('--train', dest='train', action='store_true', default=False)
parser.add_argument('--test', dest='test', action='store_true', default=False)

args = parser.parse_args()

#####################  hyper parameters  ####################

ENV_NAME = 'LunarLanderContinuous-v2'  # environment name: LunarLander-v2, Pendulum-v0
RANDOMSEED = 2  # random seed

EP_MAX = 1000  # total number of episodes for training
EP_LEN = 1000  # total number of steps for each episode
GAMMA = 0.99  # reward discount
A_LR = 0.0001  # learning rate for actor
C_LR = 0.0002  # learning rate for critic
BATCH = 4096  # update batchsize
A_UPDATE_STEPS = 50  # actor update steps
C_UPDATE_STEPS = 50  # critic update steps
HIDDEN_DIM = 64
EPS = 1e-8  # numerical residual
MODEL_PATH = 'model/ppo_multi'
NUM_WORKERS=2  # or: mp.cpu_count()
ACTION_RANGE = 1.  # normalized action range should be 1.
METHOD = [
    dict(name='kl_pen', kl_target=0.01, lam=0.5),  # KL penalty
    dict(name='clip', epsilon=0.2),  # Clipped surrogate objective
][0]  # choose the method for optimization, it's usually task specific

###############################  PPO  ####################################
class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias


class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim, init_w=3e-3):
        super(ValueNetwork, self).__init__()
        
        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        # self.linear3 = nn.Linear(hidden_dim, hidden_dim)
        self.linear4 = nn.Linear(hidden_dim, 1)

        
    def forward(self, state):
        x = F.tanh(self.linear1(state))
        x = F.tanh(self.linear2(x))
        # x = F.relu(self.linear3(x))
        x = self.linear4(x)
        return x
        
class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_range=1., init_w=3e-3, log_std_min=-20, log_std_max=2):
        super(PolicyNetwork, self).__init__()
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim)
        # self.linear4 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)

        
        # self.log_std_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std = AddBias(torch.zeros(num_actions))  

        self.num_actions = num_actions
        self.action_range = action_range

        
    def forward(self, state):
        x = F.tanh(self.linear1(state))
        x = F.tanh(self.linear2(x))
        x = F.tanh(self.linear3(x))
        # x = F.relu(self.linear4(x))

        mean    = self.action_range * F.tanh(self.mean_linear(x))
        
        # log_std = self.log_std_linear(x)
        # log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        zeros = torch.zeros(mean.size())
        if state.is_cuda:
            zeros = zeros.cuda()
        log_std = self.log_std(zeros)

        std = log_std.exp()
        return mean, std
        
    def get_action(self, state, deterministic=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        mean, std = self.forward(state)
        if deterministic:
            action = mean
        else:
            pi = torch.distributions.Normal(mean, std)
            action = pi.sample()
        action = torch.clamp(action, -self.action_range, self.action_range)
        return action.squeeze(0)

    def sample_action(self,):
        a=torch.FloatTensor(self.num_actions).uniform_(-1, 1)
        return a.numpy()

class NormalizedActions(gym.ActionWrapper):
    def _action(self, action):
        low  = self.action_space.low
        high = self.action_space.high
        
        action = low + (action + 1.0) * 0.5 * (high - low)
        action = np.clip(action, low, high)
        
        return action

    def _reverse_action(self, action):
        low  = self.action_space.low
        high = self.action_space.high
        
        action = 2 * (action - low) / (high - low) - 1
        action = np.clip(action, low, high)
        
        return action
        
class PPO(object):
    '''
    PPO class
    '''
    def __init__(self, state_dim, action_dim, hidden_dim=128, a_lr=3e-4, c_lr=3e-4):
        self.actor = PolicyNetwork(state_dim, action_dim, hidden_dim, ACTION_RANGE).to(device)
        self.critic = ValueNetwork(state_dim, hidden_dim).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=A_LR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=C_LR)
        print(self.actor, self.critic)

    def a_train(self, s, a, adv, oldpi):
        '''
        Update policy network
        :param state: state batch
        :param action: action batch
        :param adv: advantage batch
        :param old_pi: old pi distribution
        :return:
        '''  
        mu, std = self.actor(s)
        pi = Normal(mu, std)
        adv = adv.detach()  # this is critical, may not work without this line
        # ratio = torch.exp(pi.log_prob(a) - oldpi.log_prob(a))  # sometimes give nan
        ratio = torch.exp(pi.log_prob(a)) / (torch.exp(oldpi.log_prob(a)) + EPS)
        surr = ratio * adv

        if METHOD['name'] == 'kl_pen':
            lam = METHOD['lam']
            kl = torch.distributions.kl.kl_divergence(oldpi, pi)
            kl_mean = kl.mean()
            aloss = -((surr - lam * kl).mean())
        else:  # clipping method, find this is better
            aloss = -torch.mean(torch.min(surr, torch.clamp(ratio, 1. - METHOD['epsilon'], 1. + METHOD['epsilon']) * adv))
        self.actor_optimizer.zero_grad()
        aloss.backward()
        self.actor_optimizer.step()

        if METHOD['name'] == 'kl_pen':
            return kl_mean

    def c_train(self, cumulative_r, s):
        '''
        Update actor network
        :param cumulative_r: cumulative reward
        :param s: state
        :return: None
        '''
        v = self.critic(s)
        advantage = cumulative_r - v
        closs = (advantage**2).mean()
        self.critic_optimizer.zero_grad()
        closs.backward()
        self.critic_optimizer.step()

    def update(self, s, a, r):
        '''
        Update parameter with the constraint of KL divergent
        :return: None
        '''
        s = torch.Tensor(s).to(device)
        a = torch.Tensor(a).to(device)
        r = torch.Tensor(r).to(device)
        r = (r - r.mean()) / (r.std() + 1e-5)  # normalization, can be critical
        with torch.no_grad():
            mean, std = self.actor(s)
            pi = torch.distributions.Normal(mean, std)
            adv = r - self.critic(s)
        # adv = (adv - adv.mean())/(adv.std()+1e-6)  #  choose reward normalizaiton above or advantage normalization here

        # update actor
        if METHOD['name'] == 'kl_pen':
            for _ in range(A_UPDATE_STEPS):
                kl = self.a_train(s, a, adv, pi)
                if kl > 4 * METHOD['kl_target']:  # this in in google's paper
                    break
            if kl < METHOD['kl_target'] / 1.5:  # adaptive lambda, this is in OpenAI's paper
                METHOD['lam'] /= 2
            elif kl > METHOD['kl_target'] * 1.5:
                METHOD['lam'] *= 2
            METHOD['lam'] = np.clip(
                METHOD['lam'], 1e-4, 10
            )  # sometimes explode, this clipping is MorvanZhou's solution
        else:  # clipping method, find this is better (OpenAI's paper)
            for _ in range(A_UPDATE_STEPS):
                self.a_train(s, a, adv, pi)

        # update critic
        for _ in range(C_UPDATE_STEPS):
            self.c_train(r, s) 

    def choose_action(self, s, deterministic=False):
        '''
        Choose action
        :param s: state
        :return: clipped act
        '''
        a = self.actor.get_action(s, deterministic)
        return a.detach().cpu().numpy()
    
    def get_v(self, s):
        '''
        Compute value
        :param s: state
        :return: value
        '''
        s = s.astype(np.float32)
        if s.ndim < 2: s = s[np.newaxis, :]
        s = torch.FloatTensor(s).to(device)  
        return self.critic(s).squeeze(0).detach().cpu().numpy()

    def save_model(self, path):
        torch.save(self.actor.state_dict(), path+'_actor')
        torch.save(self.critic.state_dict(), path+'_critic')

    def load_model(self, path):
        self.actor.load_state_dict(torch.load(path+'_actor'))
        self.critic.load_state_dict(torch.load(path+'_critic'))

        self.actor.eval()
        self.critic.eval()

def ShareParameters(adamoptim):
    ''' share parameters of Adamoptimizers for multiprocessing '''
    for group in adamoptim.param_groups:
        for p in group['params']:
            state = adamoptim.state[p]
            # initialize: have to initialize here, or else cannot find
            state['step'] = 0
            state['exp_avg'] = torch.zeros_like(p.data)
            state['exp_avg_sq'] = torch.zeros_like(p.data)

            # share in memory
            state['exp_avg'].share_memory_()
            state['exp_avg_sq'].share_memory_()

def plot(rewards):
    clear_output(True)
    plt.figure(figsize=(10,5))
    plt.plot(rewards)
    plt.savefig('ppo_multi.png')
    # plt.show()
    plt.clf()
    plt.close()

def worker(id, ppo, rewards_queue):
    env = gym.make(ENV_NAME)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    for ep in range(EP_MAX):
        s = env.reset()
        buffer_s, buffer_a, buffer_r = [], [], []
        ep_r = 0
        t0 = time.time()
        for t in range(EP_LEN):  # in one episode
            # env.render()
            a = ppo.choose_action(s)
            s_, r, done, _ = env.step(a)
            buffer_s.append(s)
            buffer_a.append(a)
            buffer_r.append(r)
            s = s_
            ep_r += r
            # update ppo
            if (t+1) % BATCH == 0 or t == EP_LEN - 1 or done:
                if done:
                    v_s_ = 0
                else:
                    v_s_ = ppo.critic(torch.Tensor([s_]).to(device)).cpu().detach().numpy()[0, 0]
                discounted_r = []
                for r in buffer_r[::-1]:
                    v_s_ = r + GAMMA * v_s_
                    discounted_r.append(v_s_)
                discounted_r.reverse()
                bs = buffer_s if len(buffer_s[0].shape)>1 else np.vstack(buffer_s) # no vstack for raw-pixel input
                ba, br = np.vstack(buffer_a), np.array(discounted_r)[:, np.newaxis]
                buffer_s, buffer_a, buffer_r = [], [], []
                ppo.update(bs, ba, br)

            if done:
                break

        if ep%50==0:
            ppo.save_model(MODEL_PATH)
        print(
            'Episode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'.format(
                ep, EP_MAX, ep_r,
                time.time() - t0
            )
        )
        rewards_queue.put(ep_r)        
    ppo.save_model(MODEL_PATH)
    env.close()

def main():
    # reproducible
    # env.seed(RANDOMSEED)
    np.random.seed(RANDOMSEED)
    torch.manual_seed(RANDOMSEED)

    env = gym.make(ENV_NAME)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    ppo = PPO(state_dim, action_dim, hidden_dim=HIDDEN_DIM)

    if args.train:
        ppo.actor.share_memory() # this only shares memory, not the buffer for policy training
        ppo.critic.share_memory()
        ShareParameters(ppo.actor_optimizer)
        ShareParameters(ppo.critic_optimizer)
        rewards_queue=mp.Queue()  # used for get rewards from all processes and plot the curve
        processes=[]
        rewards=[]

        for i in range(NUM_WORKERS):
            process = Process(target=worker, args=(i, ppo, rewards_queue))  # the args contain shared and not shared
            process.daemon=True  # all processes closed when the main stops
            processes.append(process)

        [p.start() for p in processes]
        while True:  # keep geting the episode reward from the queue
            r = rewards_queue.get()
            if r is not None:
                if len(rewards) == 0:
                    rewards.append(r)
                else:
                    rewards.append(rewards[-1] * 0.9 + r * 0.1)
            else:
                break

            if len(rewards)%20==0 and len(rewards)>0:
                plot(rewards)

        [p.join() for p in processes]  # finished at the same time

        ppo.save_model(MODEL_PATH)
        

    if args.test:
        ppo.load_model(MODEL_PATH)
        while True:
            s = env.reset()
            eps_r=0
            for i in range(EP_LEN):
                env.render()
                s, r, done, _ = env.step(ppo.choose_action(s, True))
                eps_r+=r
                if done:
                    break
            print('Episode reward: {}  | Episode length: {}'.format(eps_r, i))
if __name__ == '__main__':
    main()
    