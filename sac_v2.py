'''
Soft Actor-Critic version 2
using target Q instead of V net: 2 Q net, 2 target Q net, 1 policy net
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

from IPython.display import clear_output
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import display
from reacher import Reacher


# use_cuda = torch.cuda.is_available()
# device   = torch.device("cuda" if use_cuda else "cpu")
# print(device)

GPU = True
device_idx = 0
if GPU:
    device = torch.device("cuda:" + str(device_idx) if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
print(device)


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = int((self.position + 1) % self.capacity)  # as a ring buffer
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch)) # stack for each element
        ''' 
        the * serves as unpack: sum(a,b) <=> batch=(a,b), sum(*batch) ;
        zip: a=[1,2], b=[2,3], zip(a,b) => [(1, 2), (2, 3)] ;
        the map serves as mapping the function on each list element: map(square, [2,3]) => [4,9] ;
        np.stack((1,2)) => array([1, 2])
        '''
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

# class NormalizedActions(gym.ActionWrapper):
#     def _action(self, action):
#         low  = self.action_space.low
#         high = self.action_space.high
        
#         action = low + (action + 1.0) * 0.5 * (high - low)
#         action = np.clip(action, low, high)
        
#         return action

#     def _reverse_action(self, action):
#         low  = self.action_space.low
#         high = self.action_space.high
        
#         action = 2 * (action - low) / (high - low) - 1
#         action = np.clip(action, low, high)
        
#         return action

def plot(frame_idx, rewards, predict_qs):
    clear_output(True)
    plt.figure(figsize=(20,5))
    # plt.subplot(131)
    plt.title('frame %s. reward: %s' % (frame_idx, rewards[-1]))
    plt.plot(rewards)
    plt.plot(predict_qs)
    plt.savefig('sac.png')
    # plt.show()

class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim, init_w=3e-3):
        super(ValueNetwork, self).__init__()
        
        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)
        # weights initialization
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x
        
        
class SoftQNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3):
        super(SoftQNetwork, self).__init__()
        
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)
        
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, state, action):
        x = torch.cat([state, action], 1) # the dim 0 is number of samples
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x
        
        
class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3, log_std_min=-20, log_std_max=2):
        super(PolicyNetwork, self).__init__()
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, hidden_size)

        self.mean_linear = nn.Linear(hidden_size, num_actions)
        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)
        
        self.log_std_linear = nn.Linear(hidden_size, num_actions)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w)

        self.action_range = 10.
        self.num_actions = num_actions

        
    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))

        mean    = (self.mean_linear(x))
        # mean    = F.leaky_relu(self.mean_linear(x))
        log_std = self.log_std_linear(x)
        # print(log_std)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std
    
    def evaluate(self, state, epsilon=1e-6):
        '''
        generate sampled action with state as input wrt the policy network;
        '''
        mean, log_std = self.forward(state)
        std = log_std.exp() # no clip in evaluation, clip affects gradients flow
        
        normal = Normal(0, 1)
        z      = normal.sample() 
        action_0 = torch.tanh(mean + std*z.to(device)) # TanhNormal distribution as actions; reparameterization trick
        action = self.action_range*action_0
        # action = F.leaky_relu(mean+ std*z.to(device))
        print('mean: ', mean[0])
        print('std: ', std[0])
        # print('action: ', action)
        # log_prob = Normal(mean, std).log_prob(mean+ std*z.to(device)) - torch.log(1. - action.pow(2) + epsilon)
        log_prob = Normal(mean, std).log_prob(mean+ std*z.to(device)) - torch.log(1. - action_0.pow(2) + epsilon) -  np.log(self.action_range)
        # both dims of normal.log_prob and -log(1-a**2) are (N,dim_of_action); 
        # the Normal.log_prob outputs the same dim of input features instead of 1 dim probability, needs sum up across the features dim to get 1 dim prob; or else use Multivariate Normal.
        print('log_prob: ', log_prob[0])
        log_prob = log_prob.sum(dim=1, keepdim=True)
        return action, log_prob, z, mean, log_std
        
    
    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        normal = Normal(0, 1)
        z      = normal.sample().to(device)
        action = self.action_range* torch.tanh(mean + std*z)
        # action = F.leaky_relu(mean+ std*z.to(device))
        
        action = action.cpu()#.detach().cpu().numpy()
        # action = action.detach().cpu()
        return action[0]


    def sample_action(self,):
        a=torch.FloatTensor(self.num_actions).uniform_(-1, 1)
        return self.action_range*a

def update(batch_size,gamma=0.99,soft_tau=1e-2,):
    alpha = 0.0  # trade-off between exploration (max entropy) and exploitation (max Q) 
    
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)
    # print('sample:', state, action,  reward, done)

    state      = torch.FloatTensor(state).to(device)
    next_state = torch.FloatTensor(next_state).to(device)
    action     = torch.FloatTensor(action).to(device)
    reward     = torch.FloatTensor(reward).unsqueeze(1).to(device)  # reward is single value, unsqueeze() to add one dim to be [reward] at the sample dim;
    done       = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)

    predicted_q_value1 = soft_q_net1(state, action)
    predicted_q_value2 = soft_q_net2(state, action)
    new_action, log_prob, epsilon, mean, log_std = policy_net.evaluate(state)
    new_next_action, next_log_prob, _, _, _ = policy_net.evaluate(next_state)

    reward = (reward - reward.mean(dim=0)) /reward.std(dim=0) # normalize with batch mean and std
# Training Q Function
    target_q_min = torch.min(target_soft_q_net1(next_state, new_next_action),target_soft_q_net2(next_state, new_next_action)) - alpha * next_log_prob
    target_q_value = reward + (1 - done) * gamma * target_q_min # if done==1, only reward
    # print('r: ', reward)
    print('pre q 1: ', predicted_q_value1[0] )
    print('t q : ', target_q_value[0])
    q_value_loss1 = soft_q_criterion1(predicted_q_value1, target_q_value.detach())  # detach: no gradients for the variable
    q_value_loss2 = soft_q_criterion2(predicted_q_value2, target_q_value.detach())


    soft_q_optimizer1.zero_grad()
    q_value_loss1.backward()
    soft_q_optimizer1.step()
    soft_q_optimizer2.zero_grad()
    q_value_loss2.backward()
    soft_q_optimizer2.step()  

# Training Policy Function
    predicted_new_q_value = torch.min(soft_q_net1(state, new_action),soft_q_net2(state, new_action))
    policy_loss = (alpha * log_prob - predicted_new_q_value).mean()

    policy_optimizer.zero_grad()
    policy_loss.backward()
    policy_optimizer.step()
    
    print('q loss: ', q_value_loss1, q_value_loss2)
    print('policy loss: ', policy_loss )


# Soft update the target value net
    for target_param, param in zip(target_soft_q_net1.parameters(), soft_q_net1.parameters()):
        target_param.data.copy_(  # copy data value into target parameters
            target_param.data * (1.0 - soft_tau) + param.data * soft_tau
        )
    for target_param, param in zip(target_soft_q_net2.parameters(), soft_q_net2.parameters()):
        target_param.data.copy_(  # copy data value into target parameters
            target_param.data * (1.0 - soft_tau) + param.data * soft_tau
        )
    return predicted_new_q_value.mean()

# intialization
# NUM_JOINTS=4
# LINK_LENGTH=[200, 140, 80, 50]
# INI_JOING_ANGLES=[0.1, 0.1, 0.1, 0.1]
NUM_JOINTS=2
LINK_LENGTH=[200, 140]
INI_JOING_ANGLES=[0.1, 0.1]
SCREEN_SIZE=1000
SPARSE_REWARD=False
SCREEN_SHOT=False
env=Reacher(screen_size=SCREEN_SIZE, num_joints=NUM_JOINTS, link_lengths = LINK_LENGTH, \
ini_joint_angles=INI_JOING_ANGLES, target_pos = [669,430], render=True)
action_dim = env.num_actions
state_dim  = env.num_observations
hidden_dim = 512

# value_net        = ValueNetwork(state_dim, hidden_dim).to(device)
# target_value_net = ValueNetwork(state_dim, hidden_dim).to(device)

soft_q_net1 = SoftQNetwork(state_dim, action_dim, hidden_dim).to(device)
soft_q_net2 = SoftQNetwork(state_dim, action_dim, hidden_dim).to(device)
target_soft_q_net1 = SoftQNetwork(state_dim, action_dim, hidden_dim).to(device)
target_soft_q_net2 = SoftQNetwork(state_dim, action_dim, hidden_dim).to(device)
policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)

print('Soft Q Network (1,2): ', soft_q_net1)
print('Policy Network: ', policy_net)

for target_param, param in zip(target_soft_q_net1.parameters(), soft_q_net1.parameters()):
    target_param.data.copy_(param.data)
for target_param, param in zip(target_soft_q_net2.parameters(), soft_q_net2.parameters()):
    target_param.data.copy_(param.data)

soft_q_criterion1 = nn.MSELoss()
soft_q_criterion2 = nn.MSELoss()

value_lr  = 3e-4
soft_q_lr = 3e-4
policy_lr = 3e-4

soft_q_optimizer1 = optim.Adam(soft_q_net1.parameters(), lr=soft_q_lr)
soft_q_optimizer2 = optim.Adam(soft_q_net2.parameters(), lr=soft_q_lr)
policy_optimizer = optim.Adam(policy_net.parameters(), lr=policy_lr)


replay_buffer_size = 1e6
replay_buffer = ReplayBuffer(replay_buffer_size)


# hyper-parameters
max_frames  = 40000
max_steps   = 20
frame_idx   = 0
batch_size  = 128
explore_steps = 200
rewards     = []
predict_qs  = []
NORM_OBS=True

# training loop
while frame_idx < max_frames:
    state = env.reset(SCREEN_SHOT)
    if NORM_OBS:
        state = state/SCREEN_SIZE
    episode_reward = 0
    predict_q = 0
    
    
    for step in range(max_steps):
        if frame_idx > explore_steps:
            action = policy_net.get_action(state).detach()
        # action = policy_net.get_action(state)
        else:
            action = policy_net.sample_action()
        print('action: ', action)
        next_state, reward, done, _ = env.step(action.numpy(), SPARSE_REWARD, SCREEN_SHOT)
        if NORM_OBS:
            next_state=next_state/SCREEN_SIZE
        replay_buffer.push(state, action, reward, next_state, done)
        
        state = next_state
        episode_reward += reward
        frame_idx += 1
        print(frame_idx)
        
        if len(replay_buffer) > batch_size:
            predict_q=update(batch_size)
            print('update')
        
        if frame_idx % batch_size == 0:
            plot(frame_idx, rewards, predict_qs)
        
        if done:
            break
        
    rewards.append(episode_reward)
    predict_qs.append(predict_q)
