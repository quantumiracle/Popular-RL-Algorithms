'''
Recurrent Deterministic Policy Gradient without target network, seems do not work!
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
from torch.utils.tensorboard import SummaryWriter
from collections import namedtuple
from common.buffers import *
from common.value_networks import *
from common.policy_networks import *

import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import display
from reacher import Reacher
import argparse
from gym import spaces


writer = SummaryWriter()
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
        self.qnet = QNetworkLSTM(state_space, action_space, hidden_dim).to(device)
        # self.target_qnet = QNetworkLSTM(state_space, action_space, hidden_dim).to(device)
        self.policy_net = DPG_PolicyNetworkLSTM(state_space, action_space, hidden_dim).to(device)
        # self.target_policy_net = DPG_PolicyNetworkLSTM(state_space, action_space, hidden_dim).to(device)

        print('Q network: ', self.qnet)
        print('Policy network: ', self.policy_net)

        # for target_param, param in zip(self.target_qnet.parameters(), self.qnet.parameters()):
        #     target_param.data.copy_(param.data)
        self.q_criterion = nn.MSELoss()
        q_lr=1e-2
        policy_lr = 1e-3
        self.update_cnt=0

        self.q_optimizer = optim.Adam(self.qnet.parameters(), lr=q_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)
    
    # def target_soft_update(self, net, target_net, soft_tau):
    # # Soft update the target net
    #     for target_param, param in zip(target_net.parameters(), net.parameters()):
    #         target_param.data.copy_(  # copy data value into target parameters
    #             target_param.data * (1.0 - soft_tau) + param.data * soft_tau
    #         )

    #     return target_net

    def update(self, batch_size, reward_scale=10.0, gamma=0.99, soft_tau=1e-2, policy_up_itr=10, target_update_delay=3, warmup=True):
        self.update_cnt+=1
        state, action, last_action, reward, next_state, done = self.replay_buffer.sample(batch_size)
        # print('sample:', state, action,  reward, done)
        q_loss = 0
        policy_loss = 0
        epi_state      = torch.FloatTensor(state).to(device)
        epi_next_state = torch.FloatTensor(next_state).to(device)
        epi_action     = torch.FloatTensor(action).to(device)
        epi_last_action     = torch.FloatTensor(last_action).to(device)
        epi_reward     = torch.FloatTensor(reward).unsqueeze(-1).to(device)  
        epi_done       = torch.FloatTensor(np.float32(done)).unsqueeze(-1).to(device)

        pi_h_out = (torch.zeros([1, batch_size, self.hidden_dim], dtype=torch.float).cuda(), \
                torch.zeros([1, batch_size, self.hidden_dim], dtype=torch.float).cuda())  # initialize hidden state for lstm, (hidden, cell), each is (layer, batch, dim)
        q_h_out = (torch.zeros([1, batch_size, self.hidden_dim], dtype=torch.float).cuda(), \
                torch.zeros([1, batch_size, self.hidden_dim], dtype=torch.float).cuda())  # initialize hidden state for lstm, (hidden, cell), each is (layer, batch, dim)
        for state, action, last_action, reward, next_state, done in zip(epi_state, epi_action, epi_last_action, epi_reward, epi_next_state, epi_done):
            state = state.unsqueeze(0) 
            action = action.unsqueeze(0)
            last_action = last_action.unsqueeze(0)
            reward = reward.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
            done       = done.unsqueeze(0)
            pi_h_in = pi_h_out
            q_h_in = q_h_out
            predict_q, q_h_out = self.qnet(state, action, last_action, q_h_in) # for q 
            new_action, pi_h_out = self.policy_net.evaluate(state, last_action, pi_h_in) # for policy
            piT_h_in = pi_h_out  #  target pi takes next state, so takes pi_h_out as piT_h_in
            # new_next_action, _ = self.target_policy_net.evaluate(next_state, action, piT_h_in)  # for q
            new_next_action, _ = self.policy_net.evaluate(next_state, action, piT_h_in)  # for q
            qT_h_in = q_h_out  # target q takes next state and next action, so takes q_h_out as qT_h_in
            # predict_target_q, _ = self.target_qnet(next_state, new_next_action, action, qT_h_in)  # for q
            predict_target_q, _ = self.qnet(next_state, new_next_action, action, qT_h_in)  # for q
            
            predict_new_q, _ = self.qnet(state, new_action, last_action, q_h_in) # for policy. as optimizers are separated, no detach for q_h_in is also fine
            target_q = reward+(1-done)*gamma*predict_target_q # for q
            # reward = reward_scale * (reward - reward.mean(dim=0)) /reward.std(dim=0) # normalize with batch mean and std

            q_loss += self.q_criterion(predict_q, target_q.detach())
            policy_loss += -torch.mean(predict_new_q)

        # train qnet
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        # train policy_net     
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

            
        # update the target_qnet
        # if self.update_cnt%target_update_delay==0:
        #     self.target_qnet=self.target_soft_update(self.qnet, self.target_qnet, soft_tau)
        #     self.target_policy_net=self.target_soft_update(self.policy_net, self.target_policy_net, soft_tau)

        return q_loss.detach().cpu().numpy(), policy_loss.detach().cpu().numpy()

    def save_model(self, path):
        torch.save(self.qnet.state_dict(), path+'_q')
        # torch.save(self.target_qnet.state_dict(), path+'_target_q')
        torch.save(self.policy_net.state_dict(), path+'_policy')

    def load_model(self, path):
        self.qnet.load_state_dict(torch.load(path+'_q'))
        # self.target_qnet.load_state_dict(torch.load(path+'_target_q'))
        self.policy_net.load_state_dict(torch.load(path+'_policy'))
        self.qnet.eval()
        # self.target_qnet.eval()
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
        # env = NormalizedActions(gym.make("Pendulum-v0"))
        env = gym.make("Pendulum-v0")
        action_space = env.action_space
        state_space  = env.observation_space
    hidden_dim = 64
    explore_steps = 0  # for random exploration
    batch_size = 1  # each sample in batch is an episode
    update_itr = 1

    replay_buffer_size=1e6
    replay_buffer = ReplayBufferLSTM(replay_buffer_size)
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
                if frame_idx > explore_steps:
                    hidden_in = hidden_out
                    action, hidden_out = alg.policy_net.get_action(state, last_action, hidden_in)
                else:
                    action = alg.policy_net.sample_action()
                next_state, reward, done, _ = env.step(action)
                if ENV !='Reacher':
                    env.render()
                
                if step>0:
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
            replay_buffer.push(episode_state, episode_action, episode_last_action, episode_reward, episode_next_state, episode_done)
            

            if i_episode % 20 == 0:
                plot(rewards)
                alg.save_model(model_path)
            if np.average(q_loss_list) is not None:
                print('Eps: ', i_episode, '| Reward: ', np.sum(episode_reward), '| Loss: ', np.average(q_loss_list), np.average(policy_loss_list))
            else:
                print('Eps: ', i_episode, '| Reward: ', np.sum(episode_reward))

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

            for step in range(max_steps):
                action = alg.policy_net.get_action(state, last_action, noise_scale=0.0)  # no noise for testing
                next_state, reward, done, _ = env.step(action)
                
                state = next_state
                episode_reward += reward
                
                
                if done:
                    break
 
            print('Eps: ', i_episode, '| Reward: ', episode_reward)
            
