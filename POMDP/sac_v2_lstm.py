'''
Soft Actor-Critic version 2
using target Q instead of V net: 2 Q net, 2 target Q net, 1 policy net
add alpha loss compared with version 1
paper: https://arxiv.org/pdf/1812.05905.pdf
'''


import math
import random

import gym
from gym_pomdp_wrappers import MuJoCoHistoryEnv
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from common.buffers import *
from common.value_networks import *
from common.policy_networks import *


from IPython.display import clear_output
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import display

import argparse
import time

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



class SAC_Trainer():
    def __init__(self, replay_buffer, state_space, action_space, hidden_dim, action_range):
        self.replay_buffer = replay_buffer

        self.soft_q_net1 = QNetworkLSTM(state_space, action_space, hidden_dim).to(device)
        self.soft_q_net2 = QNetworkLSTM(state_space, action_space, hidden_dim).to(device)
        self.target_soft_q_net1 = QNetworkLSTM(state_space, action_space, hidden_dim).to(device)
        self.target_soft_q_net2 = QNetworkLSTM(state_space, action_space, hidden_dim).to(device)
        self.policy_net = SAC_PolicyNetworkLSTM(state_space, action_space, hidden_dim, action_range).to(device)
        self.log_alpha = torch.zeros(1, dtype=torch.float32, requires_grad=True, device=device)
        print('Soft Q Network (1,2): ', self.soft_q_net1)
        print('Policy Network: ', self.policy_net)

        for target_param, param in zip(self.target_soft_q_net1.parameters(), self.soft_q_net1.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.target_soft_q_net2.parameters(), self.soft_q_net2.parameters()):
            target_param.data.copy_(param.data)

        self.soft_q_criterion1 = nn.MSELoss()
        self.soft_q_criterion2 = nn.MSELoss()

        soft_q_lr = 3e-4
        policy_lr = 3e-4
        alpha_lr  = 3e-4

        self.soft_q_optimizer1 = optim.Adam(self.soft_q_net1.parameters(), lr=soft_q_lr)
        self.soft_q_optimizer2 = optim.Adam(self.soft_q_net2.parameters(), lr=soft_q_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)

    
    def update(self, batch_size, reward_scale=10., auto_entropy=True, target_entropy=-2, gamma=0.99,soft_tau=1e-2):
        hidden_in, hidden_out, state, action, last_action, reward, next_state, done = self.replay_buffer.sample(batch_size)
        # print('sample:', state, action,  reward, done)

        state      = torch.FloatTensor(state).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        action     = torch.FloatTensor(action).to(device)
        last_action     = torch.FloatTensor(last_action).to(device)
        reward     = torch.FloatTensor(reward).unsqueeze(-1).to(device)  # reward is single value, unsqueeze() to add one dim to be [reward] at the sample dim;
        done       = torch.FloatTensor(np.float32(done)).unsqueeze(-1).to(device)

        predicted_q_value1, _ = self.soft_q_net1(state, action, last_action, hidden_in)
        predicted_q_value2, _ = self.soft_q_net2(state, action, last_action, hidden_in)
        new_action, log_prob, z, mean, log_std, _ = self.policy_net.evaluate(state, last_action, hidden_in)
        new_next_action, next_log_prob, _, _, _, _ = self.policy_net.evaluate(next_state, action, hidden_out)
        reward = reward_scale * (reward - reward.mean(dim=0)) / (reward.std(dim=0) + 1e-6) # normalize with batch mean and std; plus a small number to prevent numerical problem
    # Updating alpha wrt entropy
        # alpha = 0.0  # trade-off between exploration (max entropy) and exploitation (max Q) 
        if auto_entropy is True:
            alpha_loss = -(self.log_alpha * (log_prob + target_entropy).detach()).mean()
            # print('alpha loss: ',alpha_loss)
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()
        else:
            self.alpha = 1.
            alpha_loss = 0

    # Training Q Function
        predict_target_q1, _ = self.target_soft_q_net1(next_state, new_next_action, action, hidden_out)
        predict_target_q2, _ = self.target_soft_q_net2(next_state, new_next_action, action, hidden_out)
        target_q_min = torch.min(predict_target_q1, predict_target_q2) - self.alpha * next_log_prob
        target_q_value = reward + (1 - done) * gamma * target_q_min # if done==1, only reward
        q_value_loss1 = self.soft_q_criterion1(predicted_q_value1, target_q_value.detach())  # detach: no gradients for the variable
        q_value_loss2 = self.soft_q_criterion2(predicted_q_value2, target_q_value.detach())


        self.soft_q_optimizer1.zero_grad()
        q_value_loss1.backward()
        self.soft_q_optimizer1.step()
        self.soft_q_optimizer2.zero_grad()
        q_value_loss2.backward()
        self.soft_q_optimizer2.step()  

    # Training Policy Function
        predict_q1, _= self.soft_q_net1(state, new_action, last_action, hidden_in)
        predict_q2, _ = self.soft_q_net2(state, new_action, last_action, hidden_in)
        predicted_new_q_value = torch.min(predict_q1, predict_q2)
        policy_loss = (self.alpha * log_prob - predicted_new_q_value).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

    # Soft update the target value net
        for target_param, param in zip(self.target_soft_q_net1.parameters(), self.soft_q_net1.parameters()):
            target_param.data.copy_(  # copy data value into target parameters
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )
        for target_param, param in zip(self.target_soft_q_net2.parameters(), self.soft_q_net2.parameters()):
            target_param.data.copy_(  # copy data value into target parameters
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )
        return predicted_new_q_value.mean()

    def save_model(self, path):
        torch.save(self.soft_q_net1.state_dict(), path+'_q1')
        torch.save(self.soft_q_net2.state_dict(), path+'_q2')
        torch.save(self.policy_net.state_dict(), path+'_policy')

    def load_model(self, path):
        self.soft_q_net1.load_state_dict(torch.load(path+'_q1'))
        self.soft_q_net2.load_state_dict(torch.load(path+'_q2'))
        self.policy_net.load_state_dict(torch.load(path+'_policy'))

        self.soft_q_net1.eval()
        self.soft_q_net2.eval()
        self.policy_net.eval()


def plot(rewards):
    clear_output(True)
    plt.figure(figsize=(20,5))
    plt.plot(rewards)
    plt.savefig('sac_v2_lstm.png')
    # plt.show()


replay_buffer_size = 1e6
replay_buffer = ReplayBufferLSTM2(replay_buffer_size)

# choose env
ENV = 'HalfCheetah-v2'
env = NormalizedActions(MuJoCoHistoryEnv(ENV, hist_len=0, history_type="pomdp"))
action_space = env.action_space
state_space  = env.observation_space
action_range=1.

action_dim = action_space.shape[0]

# hyper-parameters for RL training
max_episodes  = 1000
max_steps   = 150  
frame_idx   = 0
batch_size  = 2
explore_steps = 0  # for action sampling in the beginning of training
update_itr = 1
AUTO_ENTROPY=True
DETERMINISTIC=False
hidden_dim =512
rewards     = []
model_path = './model/sac_v2_lstm'

sac_trainer=SAC_Trainer(replay_buffer, state_space, action_space, hidden_dim=hidden_dim, action_range=action_range  )

if __name__ == '__main__':
    if args.train:
        # training loop
        for eps in range(max_episodes):
            if ENV == 'Reacher':
                state = env.reset(SCREEN_SHOT)
            else:
                state =  env.reset()
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
                action, hidden_out = sac_trainer.policy_net.get_action(state, last_action, hidden_in, deterministic = DETERMINISTIC)
                if ENV ==  'Reacher':
                    next_state, reward, done, _ = env.step(action, SPARSE_REWARD, SCREEN_SHOT)
                else:
                    next_state, reward, done, _ = env.step(action)
                    # env.render()       
                    
                if step>0:
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
                    for i in range(update_itr):
                        _=sac_trainer.update(batch_size, reward_scale=10., auto_entropy=AUTO_ENTROPY, target_entropy=-1.*action_dim)

                if done:
                    break
            replay_buffer.push(ini_hidden_in, ini_hidden_out, episode_state, episode_action, episode_last_action, \
                episode_reward, episode_next_state, episode_done)

            if eps % 20 == 0 and eps>0: # plot and model saving interval
                plot(rewards)
                np.save('rewards_lstm', rewards)
                sac_trainer.save_model(model_path)
            print('Episode: ', eps, '| Episode Reward: ', np.sum(episode_reward))
            rewards.append(np.sum(episode_reward))
        sac_trainer.save_model(model_path)

    if args.test:
        sac_trainer.load_model(model_path)
        for eps in range(10):
            if ENV == 'Reacher':
                state = env.reset(SCREEN_SHOT)
            else:
                state =  env.reset()
            last_action = env.action_space.sample()
            episode_reward = 0
            hidden_out = (torch.zeros([1, 1, hidden_dim], dtype=torch.float).cuda(), \
                torch.zeros([1, 1, hidden_dim], dtype=torch.float).cuda())  # initialize hidden state for lstm, (hidden, cell), each is (layer, batch, dim)
            
            for step in range(max_steps):
                hidden_in = hidden_out
                action, hidden_out = sac_trainer.policy_net.get_action(state, last_action, hidden_in, deterministic = DETERMINISTIC)
                if ENV ==  'Reacher':
                    next_state, reward, done, _ = env.step(action, SPARSE_REWARD, SCREEN_SHOT)
                else:
                    next_state, reward, done, _ = env.step(action)
                    env.render()   

                last_action = action
                episode_reward += reward
                state=next_state

            print('Episode: ', eps, '| Episode Reward: ', episode_reward)
