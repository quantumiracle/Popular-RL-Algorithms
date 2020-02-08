'''
Twin Delayed DDPG (TD3), if no twin no delayed then it's DDPG.
using target Q instead of V net: 2 Q net, 2 target Q net, 1 policy net, 1 target policy net
original paper: https://arxiv.org/pdf/1802.09477.pdf
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
from common.buffers import *
from common.value_networks import *
from common.policy_networks import *

from IPython.display import clear_output
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import display
from reacher import Reacher
import gym.spaces as spaces

import argparse
import time

torch.manual_seed(1234)  #Reproducibility

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
        

class TD3_Trainer():
    def __init__(self, replay_buffer, state_space, action_space, hidden_dim, action_range, policy_target_update_interval=1):
        self.replay_buffer = replay_buffer
        self.hidden_dim = hidden_dim

        self.q_net1 = QNetworkLSTM(state_space, action_space, hidden_dim).to(device)
        self.q_net2 = QNetworkLSTM(state_space, action_space, hidden_dim).to(device)
        self.target_q_net1 = QNetworkLSTM(state_space, action_space, hidden_dim).to(device)
        self.target_q_net2 = QNetworkLSTM(state_space, action_space, hidden_dim).to(device)
        self.policy_net = DPG_PolicyNetworkLSTM(state_space, action_space, hidden_dim).to(device)
        self.target_policy_net = DPG_PolicyNetworkLSTM(state_space, action_space, hidden_dim).to(device)
        print('Q Network (1,2): ', self.q_net1)
        print('Policy Network: ', self.policy_net)

        self.target_q_net1 = self.target_ini(self.q_net1, self.target_q_net1)
        self.target_q_net2 = self.target_ini(self.q_net2, self.target_q_net2)
        self.target_policy_net = self.target_ini(self.policy_net, self.target_policy_net)
        
        q_lr = 3e-4
        policy_lr = 3e-4
        self.update_cnt = 0
        self.policy_target_update_interval = policy_target_update_interval

        self.q_optimizer1 = optim.Adam(self.q_net1.parameters(), lr=q_lr)
        self.q_optimizer2 = optim.Adam(self.q_net2.parameters(), lr=q_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)
    
    def target_ini(self, net, target_net):
        for target_param, param in zip(target_net.parameters(), net.parameters()):
            target_param.data.copy_(param.data)
        return target_net

    def target_soft_update(self, net, target_net, soft_tau):
    # Soft update the target net
        for target_param, param in zip(target_net.parameters(), net.parameters()):
            target_param.data.copy_(  # copy data value into target parameters
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )

        return target_net
    
    def update(self, batch_size, deterministic, eval_noise_scale, reward_scale=10., gamma=0.9, soft_tau=1e-2):
        hidden_in, hidden_out, state, action, last_action, reward, next_state, done = self.replay_buffer.sample(batch_size)
        # print('sample:', state, action,  reward, done)
        state      = torch.FloatTensor(state).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        action     = torch.FloatTensor(action).to(device)
        last_action     = torch.FloatTensor(last_action).to(device)
        reward     = torch.FloatTensor(reward).unsqueeze(-1).to(device)  
        done       = torch.FloatTensor(np.float32(done)).unsqueeze(-1).to(device)
 
        predicted_q_value1, _ = self.q_net1(state, action, last_action, hidden_in)
        predicted_q_value2, _ = self.q_net2(state, action, last_action, hidden_in)
        new_action,  _= self.policy_net.evaluate(state, last_action, hidden_in, noise_scale=0.0)  # no noise, deterministic policy gradients
        new_next_action, _ = self.target_policy_net.evaluate(next_state, action, hidden_out, noise_scale=eval_noise_scale) # clipped normal noise
        # reward = reward_scale * (reward - reward.mean(dim=0)) / (reward.std(dim=0) + 1e-6) # normalize with batch mean and std; plus a small number to prevent numerical problem

        # Training Q Function
        predicted_target_q1, _ = self.target_q_net1(next_state, new_next_action, action, hidden_out)
        predicted_target_q2, _ = self.target_q_net2(next_state, new_next_action, action, hidden_out)
        target_q_min = torch.min(predicted_target_q1, predicted_target_q2)

        target_q_value = reward + (1 - done) * gamma * target_q_min # if done==1, only reward

        q_value_loss1 = ((predicted_q_value1 - target_q_value.detach())**2).mean()  # detach: no gradients for the variable
        q_value_loss2 = ((predicted_q_value2 - target_q_value.detach())**2).mean()         
        self.q_optimizer1.zero_grad()
        q_value_loss1.backward()
        self.q_optimizer1.step()
        self.q_optimizer2.zero_grad()
        q_value_loss2.backward()
        self.q_optimizer2.step()
        
        if self.update_cnt%self.policy_target_update_interval==0:
            # Training Policy Function
            ''' implementation 1 '''
            # predicted_new_q_value = torch.min(self.q_net1(state, new_action),self.q_net2(state, new_action))
            ''' implementation 2 '''
            predicted_new_q_value, _ = self.q_net1(state, new_action, last_action, hidden_in)

            policy_loss = - predicted_new_q_value.mean()
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()
            # Soft update the target nets
            self.target_q_net1=self.target_soft_update(self.q_net1, self.target_q_net1, soft_tau)
            self.target_q_net2=self.target_soft_update(self.q_net2, self.target_q_net2, soft_tau)
            self.target_policy_net=self.target_soft_update(self.policy_net, self.target_policy_net, soft_tau)

        self.update_cnt+=1

        return predicted_q_value1.mean()  # for debug

    def save_model(self, path):
        torch.save(self.q_net1.state_dict(), path+'_q1')
        torch.save(self.q_net2.state_dict(), path+'_q2')
        torch.save(self.policy_net.state_dict(), path+'_policy')

    def load_model(self, path):
        self.q_net1.load_state_dict(torch.load(path+'_q1'))
        self.q_net2.load_state_dict(torch.load(path+'_q2'))
        self.policy_net.load_state_dict(torch.load(path+'_policy'))
        self.q_net1.eval()
        self.q_net2.eval()
        self.policy_net.eval()

def plot(rewards):
    clear_output(True)
    plt.figure(figsize=(20,5))
    plt.plot(rewards)
    plt.savefig('td3_lstm.png')
    # plt.show()


# choose env
ENV = ['Reacher', 'Pendulum-v0', 'HalfCheetah-v2'][1]
if ENV == 'Reacher':
    NUM_JOINTS=2
    LINK_LENGTH=[200, 140]
    INI_JOING_ANGLES=[0.1, 0.1]
    SCREEN_SIZE=1000
    SPARSE_REWARD=False
    SCREEN_SHOT=False
    action_range = 10.0

    env=Reacher(screen_size=SCREEN_SIZE, num_joints=NUM_JOINTS, link_lengths = LINK_LENGTH, \
    ini_joint_angles=INI_JOING_ANGLES, target_pos = [369,430], render=True, change_goal=False)
    action_space = spaces.Box(low=-1.0, high=1.0, shape=(env.num_actions,), dtype=np.float32)
    state_space  = spaces.Box(low=-np.inf, high=np.inf, shape=(env.num_observations, ))
else:
    env = NormalizedActions(gym.make(ENV))
    action_space = env.action_space
    state_space  = env.observation_space
    action_range=1.

replay_buffer_size = 5e5
replay_buffer = ReplayBufferLSTM2(replay_buffer_size)


# hyper-parameters for RL training
max_episodes  = 1000
max_steps   = 20 if ENV ==  'Reacher' else 150  # Pendulum needs 150 steps per episode to learn well, cannot handle 20
frame_idx   = 0
batch_size  = 2  # each sample contains an episode for lstm policy
explore_steps = 0  # for random action sampling in the beginning of training
update_itr = 1
hidden_dim = 512
policy_target_update_interval = 10 # delayed update for the policy network and target networks
DETERMINISTIC=True  # DDPG: deterministic policy gradient
explore_noise_scale = 0.5
eval_noise_scale = 0.5
reward_scale = 1.
rewards     = []
model_path = './model/td3_lstm'

td3_trainer=TD3_Trainer(replay_buffer, state_space, action_space, hidden_dim=hidden_dim, \
    policy_target_update_interval=policy_target_update_interval, action_range=action_range )

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
                action, hidden_out = td3_trainer.policy_net.get_action(state, last_action, hidden_in, noise_scale=explore_noise_scale)
                if ENV ==  'Reacher':
                    next_state, reward, done, _ = env.step(action, SPARSE_REWARD, SCREEN_SHOT)
                else:
                    next_state, reward, done, _ = env.step(action) 
                    # env.render()
                if step == 0:
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
                        _=td3_trainer.update(batch_size, deterministic=DETERMINISTIC, eval_noise_scale=eval_noise_scale, reward_scale=reward_scale)
                
                if done:
                    break
            
            replay_buffer.push(ini_hidden_in, ini_hidden_out, episode_state, episode_action, episode_last_action, \
                episode_reward, episode_next_state, episode_done)
            if eps % 20 == 0 and eps>0:
                plot(rewards)
                np.save('rewards_td3_lstm', rewards)
                td3_trainer.save_model(model_path)

            print('Episode: ', eps, '| Episode Reward: ', np.sum(episode_reward))
            rewards.append(np.sum(episode_reward))
        td3_trainer.save_model(model_path)
        
    if args.test:
        td3_trainer.load_model(model_path)
        for eps in range(10):
            if ENV == 'Reacher':
                state = env.reset(SCREEN_SHOT)
            else:
                state =  env.reset()
                env.render()   
            last_action = env.action_space.sample()
            episode_reward = 0
            hidden_out = (torch.zeros([1, 1, hidden_dim], dtype=torch.float).cuda(), \
                torch.zeros([1, 1, hidden_dim], dtype=torch.float).cuda())  # initialize hidden state for lstm, (hidden, cell), each is (layer, batch, dim)
            
            for step in range(max_steps):
                hidden_in = hidden_out
                action, hidden_out = td3_trainer.policy_net.get_action(state, last_action, hidden_in, noise_scale=0.)
                if ENV ==  'Reacher':
                    next_state, reward, done, _ = env.step(action, SPARSE_REWARD, SCREEN_SHOT)
                else:
                    next_state, reward, done, _ = env.step(action)
                    env.render() 

                last_action = action
                episode_reward += reward
                state=next_state

            print('Episode: ', eps, '| Episode Reward: ', episode_reward)
