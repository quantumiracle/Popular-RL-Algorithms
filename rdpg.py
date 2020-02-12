'''
Recurrent Deterministic Policy Gradient (DDPG with LSTM network)
Update with batch of episodes for each time, so requires each episode has the same length.
'''


import math
import random

import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions import Categorical
from collections import namedtuple
from common.buffers import *
from common.value_networks import *
from common.policy_networks import *
from common.utils import *

import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import display
from reacher import Reacher
import argparse
from gym import spaces


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

class RDPG():
    def __init__(self, replay_buffer, state_space, action_space, hidden_dim):
        self.replay_buffer = replay_buffer
        self.hidden_dim = hidden_dim
        # single-branch network structure as in 'Memory-based control with recurrent neural networks'
        self.qnet = QNetworkLSTM2(state_space, action_space, hidden_dim).to(device)
        self.target_qnet = QNetworkLSTM2(state_space, action_space, hidden_dim).to(device)
        self.policy_net = DPG_PolicyNetworkLSTM2(state_space, action_space, hidden_dim).to(device)
        self.target_policy_net = DPG_PolicyNetworkLSTM2(state_space, action_space, hidden_dim).to(device)

        # two-branch network structure as in 'Sim-to-Real Transfer of Robotic Control with Dynamics Randomization'
        # self.qnet = QNetworkLSTM(state_space, action_space, hidden_dim).to(device)
        # self.target_qnet = QNetworkLSTM(state_space, action_space, hidden_dim).to(device)
        # self.policy_net = DPG_PolicyNetworkLSTM(state_space, action_space, hidden_dim).to(device)
        # self.target_policy_net = DPG_PolicyNetworkLSTM(state_space, action_space, hidden_dim).to(device)

        print('Q network: ', self.qnet)
        print('Policy network: ', self.policy_net)

        for target_param, param in zip(self.target_qnet.parameters(), self.qnet.parameters()):
            target_param.data.copy_(param.data)
        self.q_criterion = nn.MSELoss()
        q_lr=1e-3
        policy_lr = 1e-3
        self.update_cnt=0

        self.q_optimizer = optim.Adam(self.qnet.parameters(), lr=q_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)
    
    def target_soft_update(self, net, target_net, soft_tau):
    # Soft update the target net
        for target_param, param in zip(target_net.parameters(), net.parameters()):
            target_param.data.copy_(  # copy data value into target parameters
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )

        return target_net

    def update(self, batch_size, reward_scale=10.0, gamma=0.99, soft_tau=1e-2, policy_up_itr=10, target_update_delay=3, warmup=True):
        self.update_cnt+=1
        hidden_in, hidden_out, state, action, last_action, reward, next_state, done = self.replay_buffer.sample(batch_size)
        # print('sample:', state, action,  reward, done)
        state      = torch.FloatTensor(state).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        action     = torch.FloatTensor(action).to(device)
        last_action     = torch.FloatTensor(last_action).to(device)
        reward     = torch.FloatTensor(reward).unsqueeze(-1).to(device)  
        done       = torch.FloatTensor(np.float32(done)).unsqueeze(-1).to(device)

        # use hidden states stored in the memory for initialization, hidden_in for current, hidden_out for target
        predict_q, _ = self.qnet(state, action, last_action, hidden_in) # for q 
        new_action, _ = self.policy_net.evaluate(state, last_action, hidden_in) # for policy
        new_next_action, _ = self.target_policy_net.evaluate(next_state, action, hidden_out)  # for q
        predict_target_q, _ = self.target_qnet(next_state, new_next_action, action, hidden_out)  # for q

        predict_new_q, _ = self.qnet(state, new_action, last_action, hidden_in) # for policy. as optimizers are separated, no detach for q_h_in is also fine
        target_q = reward+(1-done)*gamma*predict_target_q # for q
        # reward = reward_scale * (reward - reward.mean(dim=0)) /reward.std(dim=0) # normalize with batch mean and std

        q_loss = self.q_criterion(predict_q, target_q.detach())
        policy_loss = -torch.mean(predict_new_q)

        # train qnet
        self.q_optimizer.zero_grad()
        q_loss.backward(retain_graph=True)  # no need for retain_graph here actually
        self.q_optimizer.step()

        # train policy_net     
        self.policy_optimizer.zero_grad()
        policy_loss.backward(retain_graph=True)
        self.policy_optimizer.step()

            
        # update the target_qnet
        if self.update_cnt%target_update_delay==0:
            self.target_qnet=self.target_soft_update(self.qnet, self.target_qnet, soft_tau)
            self.target_policy_net=self.target_soft_update(self.policy_net, self.target_policy_net, soft_tau)

        return q_loss.detach().cpu().numpy(), policy_loss.detach().cpu().numpy()

    def save_model(self, path):
        torch.save(self.qnet.state_dict(), path+'_q')
        torch.save(self.target_qnet.state_dict(), path+'_target_q')
        torch.save(self.policy_net.state_dict(), path+'_policy')

    def load_model(self, path):
        self.qnet.load_state_dict(torch.load(path+'_q'))
        self.target_qnet.load_state_dict(torch.load(path+'_target_q'))
        self.policy_net.load_state_dict(torch.load(path+'_policy'))
        self.qnet.eval()
        self.target_qnet.eval()
        self.policy_net.eval()

def plot(rewards):
    plt.figure(figsize=(20,5))
    plt.plot(rewards)
    plt.savefig('rdpg.png')
    # plt.show()
    plt.clf()

class NormalizedActions(gym.ActionWrapper): # gym env wrapper
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


if __name__ == '__main__':
    NUM_JOINTS=2
    LINK_LENGTH=[200, 140]
    INI_JOING_ANGLES=[0.1, 0.1]
    SCREEN_SIZE=1000
    # SPARSE_REWARD=False
    # SCREEN_SHOT=False
    ENV = ['Pendulum', 'Reacher'][0]
    if ENV == 'Reacher':
        env=Reacher(screen_size=SCREEN_SIZE, num_joints=NUM_JOINTS, link_lengths = LINK_LENGTH, \
        ini_joint_angles=INI_JOING_ANGLES, target_pos = [369,430], render=True)
        action_space = spaces.Box(low=-1.0, high=1.0, shape=(env.num_actions,), dtype=np.float32)
        state_space  = spaces.Box(low=-np.inf, high=np.inf, shape=(env.num_observations, ))

    elif ENV == 'Pendulum':
        env = NormalizedActions(gym.make("Pendulum-v0"))
        # env = gym.make("Pendulum-v0")
        action_space = env.action_space
        state_space  = env.observation_space
    hidden_dim = 64
    explore_steps = 0  # for random exploration
    batch_size = 3  # each sample in batch is an episode for lstm policy (normally it's timestep)
    update_itr = 1  # update iteration

    replay_buffer_size=1e6
    replay_buffer = ReplayBufferLSTM2(replay_buffer_size)
    model_path='./model/rdpg'
    torch.autograd.set_detect_anomaly(True)
    alg = RDPG(replay_buffer, state_space, action_space, hidden_dim)

    if args.train:
        # alg.load_model(model_path)

        # hyper-parameters
        max_episodes  = 1000
        max_steps   = 100
        frame_idx   = 0
        rewards=[]

        for i_episode in range (max_episodes):
            q_loss_list=[]
            policy_loss_list=[]
            state = env.reset()
            episode_reward = 0
            last_action = env.action_space.sample()
            episode_state = []
            episode_action = []
            episode_last_action = []
            episode_reward = []
            episode_next_state = []
            episode_done = []
            hidden_out = (torch.zeros([1, 1, hidden_dim], dtype=torch.float).cuda(), \
                torch.zeros([1, 1, hidden_dim], dtype=torch.float).cuda())  # initialize hidden state for lstm, (hidden, cell), each is (layer, batch, dim)
            
            for step in range(max_steps):
                hidden_in = hidden_out
                action, hidden_out = alg.policy_net.get_action(state, last_action, hidden_in)

                next_state, reward, done, _ = env.step(action)
                if ENV !='Reacher':
                    env.render()
                if step==0:
                    ini_hidden_in = hidden_in
                    ini_hidden_out = hidden_out
                episode_state.append(state)
                episode_action.append(action)
                episode_last_action.append(last_action)
                episode_reward.append(reward)
                episode_next_state.append(next_state)
                episode_done.append(done)  

                state = next_state
                last_action = action
                frame_idx += 1
                if len(replay_buffer) > batch_size:
                    for _ in range(update_itr):
                        q_loss, policy_loss = alg.update(batch_size)
                        q_loss_list.append(q_loss)
                        policy_loss_list.append(policy_loss)
                if done:  # should not break for lstm cases to make every episode with same length
                    break        

            if i_episode % 20 == 0:
                plot(rewards)
                alg.save_model(model_path)
            print('Eps: ', i_episode, '| Reward: ', np.sum(episode_reward), '| Loss: ', np.average(q_loss_list), np.average(policy_loss_list))
            replay_buffer.push(ini_hidden_in, ini_hidden_out, episode_state, episode_action, episode_last_action, \
                episode_reward, episode_next_state, episode_done)

            rewards.append(np.sum(episode_reward))
            alg.save_model(model_path)


    if args.test:
        test_episodes = 10
        max_steps=100
        alg.load_model(model_path)

        for i_episode in range (test_episodes):
            q_loss_list=[]
            policy_loss_list=[]
            state = env.reset()
            episode_reward = 0
            last_action = np.zeros(action_space.shape[0])
            hidden_out = (torch.zeros([1, 1, hidden_dim], dtype=torch.float).cuda(), \
                torch.zeros([1, 1, hidden_dim], dtype=torch.float).cuda())  # initialize hidden state for lstm, (hidden, cell), each is (layer, batch, dim)
            
            for step in range(max_steps):
                hidden_in = hidden_out
                action, hidden_out= alg.policy_net.get_action(state, last_action, hidden_in, noise_scale=0.0)  # no noise for testing
                next_state, reward, done, _ = env.step(action)
                env.render()
                
                last_action = action
                state = next_state
                episode_reward += reward
                
                
                if done:
                    break
 
            print('Eps: ', i_episode, '| Reward: ', episode_reward)
            
