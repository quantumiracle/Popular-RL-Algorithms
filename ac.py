'''
Actor-Critic, actually Advantage Actor-Critic (A2C).

Policy loss in Vanilla Actor-Critic is: -log_prob(a)*Q(s,a) ,
Policy loss in A2C is: -log_prob(a)*[Q(s,a)-V(s)], while Adv(s,a)=Q(s,a)-V(s)=r+gamma*V(s')-V(s)=TD_error ,
and in this implementation we provide another approach that the V(s') is replaced by R(s'), 
which is derived from the rewards in the episode for on-policy update without evaluation. 

Discrete and Non-deterministic
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

DISCRETE = True # discrete actions if ture, else continuous
DETERMINISTIC = False # deterministic actions if true, like DDPG or DQN's argmax, else non-deterministic (sampling)
if DISCRETE:
    # each output node corresponds to one possible action, 
    # the output dim = possible action values (only one action)
    pass
else: 
    # the output dim = dim of action
    pass
if DETERMINISTIC:
    # no need of sampling, directly output actions
    pass
else:
    # output the mean and (log-)variance for Gaussian prior, then sampling 
    pass
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


class ActorNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, init_w=3e-3):
        super(ActorNetwork, self).__init__()

        self.saved_logprobs = [] # this is critical! have to save the values inside the model to keep track of its gradients
        self.saved_entropies = []

        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        
        if DISCRETE: # e.g. DQN for deterministic and Actor-Critic for non-deterministic
            self.linear3 = nn.Linear(hidden_dim, output_dim) # output dim = possible action values

            # weights initialization
            self.linear3.weight.data.uniform_(-init_w, init_w)
            self.linear3.bias.data.uniform_(-init_w, init_w)
            
        elif not DISCRETE and DETERMINISTIC: # e.g. DDPG
            self.linear3 = nn.Linear(hidden_dim, output_dim) # output dim = dim of action

            # weights initialization
            self.linear3.weight.data.uniform_(-init_w, init_w)
            self.linear3.bias.data.uniform_(-init_w, init_w)
        
        elif not DISCRETE and not DETERMINISTIC: # e.g. REINFORCE, Actor-Critic, PPO for continuous case
            self.mean_linear = nn.Linear(hidden_dim, output_dim) # output dim = dim of action
            self.log_std_linear = nn.Linear(hidden_dim, output_dim)

            # weights initialization
            self.mean_linear.weight.data.uniform_(-init_w, init_w)
            self.mean_linear.bias.data.uniform_(-init_w, init_w)
            self.log_std_linear.weight.data.uniform_(-init_w, init_w)
            self.log_std_linear.bias.data.uniform_(-init_w, init_w)

   
    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        if DISCRETE and DETERMINISTIC:
            x = torch.max(self.linear3(x), dim=-1)
            return x
        elif DISCRETE and not DETERMINISTIC:
            x = F.softmax(self.linear3(x), dim=-1)
            return x
        elif not DISCRETE and not DETERMINISTIC:
            self.log_std_min=-20
            self.log_std_max=2

            mean = self.mean_linear(x)
            log_std = self.log_std_linear(x)
            log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
            return mean, log_std
        else:
            x = self.linear3(x)
            return x

    def select_action(self, state):
        '''
        only select action without the purpose of gradients flow, for interaction with env to
        generate samples
        '''
        if DETERMINISTIC:
            action = self.forward(state)

        if DISCRETE and not DETERMINISTIC:
            probs = self.forward(state)
            m = Categorical(probs)
            action = m.sample()

        if not DISCRETE and not DETERMINISTIC:
            self.action_range = 30.

            mean, log_std = self.forward(state)
            std = log_std.exp()
            normal = Normal(0, 1)
            z = normal.sample().to(device)
            action = self.action_range* torch.tanh(mean + std*z)
 
        return action.detach()


    def evaluate_action(self, state):
        '''
        evaluate action within GPU graph, for gradients flowing through it
        '''
        state = torch.FloatTensor(state).unsqueeze(0).to(device) # state dim: (N, dim of state)
        if DETERMINISTIC:
            action = self.forward(state)
            return action.detach().cpu().numpy()

        elif DISCRETE and not DETERMINISTIC:  # actor-critic (discrete)
            probs = self.forward(state)
            m = Categorical(probs)
            action = m.sample().to(device)
            log_prob = m.log_prob(action)

            return action.detach().cpu().numpy(), log_prob.squeeze(0), m.entropy().mean()

        elif not DISCRETE and not DETERMINISTIC: # soft actor-critic (continuous)
            self.action_range = 30.
            self.epsilon = 1e-6

            mean, log_std = self.forward(state)
            std = log_std.exp()
            normal = Normal(0, 1)
            z = normal.sample().to(device)
            action0 = torch.tanh(mean + std*z.to(device)) # TanhNormal distribution as actions; reparameterization trick
            action = self.action_range * action0
            
            log_prob = Normal(mean, std).log_prob(mean+ std*z.to(device)) - torch.log(1. - action0.pow(2) + self.epsilon) -  np.log(self.action_range)            
            log_prob = log_prob.sum(dim=1, keepdim=True)
            # print('mean: ', mean, 'log_std: ', log_std)
            # return action.item(), log_prob, z, mean, log_std
            return action.detach().cpu().numpy().squeeze(0), log_prob.squeeze(0), Normal(mean, std).entropy().mean()

class CriticNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, init_w=3e-3):
        super(CriticNetwork, self).__init__()

        self.saved_values = [] # this is critical! have to save the values inside the model to keep track of its gradients
        self.saved_nextvalues = []
    
        
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)
        # weights initialization
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, state):
        state = torch.FloatTensor(state).to(device)
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


# class QNetwork(nn.Module):
#     def __init__(self, input_dim, hidden_dim, init_w=3e-3):
#         super(QNetwork, self).__init__()
        
#         self.linear1 = nn.Linear(input_dim, hidden_dim)
#         self.linear2 = nn.Linear(hidden_dim, hidden_dim)
#         self.linear3 = nn.Linear(hidden_dim, 1)
        
#         self.linear3.weight.data.uniform_(-init_w, init_w)
#         self.linear3.bias.data.uniform_(-init_w, init_w)
        
#     def forward(self, state, action):
#         x = torch.cat([state, action], 1) # the dim 0 is number of samples
#         x = F.relu(self.linear1(x))
#         x = F.relu(self.linear2(x))
#         x = self.linear3(x)
#         return x

def Update0(rewards, gamma=0.99, entropy_lambda=1e-3): 
    ''' update with R(s') instead of V(s') in the TD-error;
        with entropy boosting exploration
    '''
    # print('sets: ', actions)
    # print('rewards: ', rewards)
    R = 0
    policy_losses = []
    value_losses = []
    rewards_ = []
    eps = np.finfo(np.float32).eps.item()

    for r in rewards[::-1]:
        R = r + gamma * R
        rewards_.insert(0, R)
    rewards_ = torch.tensor(rewards_).to(device)
    rewards_ = (rewards_ - rewards_.mean()) / (rewards_.std() + eps)
    # print('rewards: ', rewards)
    # print('rewards_: ', rewards_)
    for log_prob, value, r in zip(actor_net.saved_logprobs, critic_net.saved_values, rewards_):
        value_losses.append(F.smooth_l1_loss(value, torch.tensor([r]).to(device)))
        td_error = r - value.detach().item() # value gradients flow only through the critic
        policy_losses.append(-log_prob * td_error)
    # print('policy losses: ', policy_losses)
    # print('value losses: ', value_losses)
    actor_optimizer.zero_grad()
    policy_loss=torch.stack(policy_losses).sum() - entropy_lambda * torch.stack(actor_net.saved_entropies).sum()
    policy_loss.backward()
    actor_optimizer.step()
    critic_optimizer.zero_grad()
    value_loss=torch.stack(value_losses).sum()
    value_loss.backward()
    # print('loss: ', policy_loss, value_loss)
    critic_optimizer.step()
    del actor_net.saved_logprobs[:]
    del critic_net.saved_values[:]
    del actor_net.saved_entropies[:]




def Update1(rewards, gamma=0.99): 
    ''' update with V(s') in the TD-error'''
    policy_losses = []
    value_losses = []
    value_criterion  = nn.MSELoss()

    rewards = torch.tensor(rewards).to(device)
    for log_prob, state_value, next_state_value, r in zip(actor_net.saved_logprobs, critic_net.saved_values, critic_net.saved_nextvalues, rewards):
        # value_losses.append(F.smooth_l1_loss(state_value, r + gamma * next_state_value.detach_())) # detach the next_state_value, only BP through state_value
        value_losses.append(value_criterion(state_value, r + gamma * next_state_value.detach_()))
        state_value.detach_() # detach in place
        policy_losses.append(-log_prob * (r + gamma * next_state_value - state_value)) # only BP through the log_prob for actor update
    # print('policy losses: ', policy_losses)
    # print('value losses: ', value_losses)
    actor_optimizer.zero_grad()
    policy_loss=torch.stack(policy_losses).sum()
    policy_loss.backward()
    actor_optimizer.step()
    critic_optimizer.zero_grad()
    value_loss=torch.stack(value_losses).sum()
    value_loss.backward()
    # print('loss: ', policy_loss, value_loss)
    critic_optimizer.step()
    del actor_net.saved_logprobs[:]
    del critic_net.saved_values[:]
    del critic_net.saved_nextvalues[:]



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


def plot(frame_idx, rewards):
    clear_output(True)
    plt.figure(figsize=(20,5))
    # plt.subplot(131)
    plt.title('frame %s. reward: %s' % (frame_idx, rewards[-1]))
    plt.plot(rewards)
    # plt.plot(predict_qs)
    plt.savefig('ac.png')
    # plt.show()


ON_POLICY=True
hidden_dim = 30   
UPDATE=['Approach0', 'Approach1'][0]
# choose env
ENV = ['Pendulum-v0', 'CartPole-v0', 'Reacher'][1]  # Pendulum is continuous, CartPole is discrete
if ENV == 'Reacher':
    DISCRETE = False
    hidden_dim = 512
    NUM_JOINTS=2
    LINK_LENGTH=[200, 140]
    INI_JOING_ANGLES=[0.1, 0.1]
    # NUM_JOINTS=4
    # LINK_LENGTH=[200, 140, 80, 50]
    # INI_JOING_ANGLES=[0.1, 0.1, 0.1, 0.1]
    SCREEN_SIZE=1000
    SPARSE_REWARD=False
    SCREEN_SHOT=False
    action_range = 10.0

    env=Reacher(screen_size=SCREEN_SIZE, num_joints=NUM_JOINTS, link_lengths = LINK_LENGTH, \
    ini_joint_angles=INI_JOING_ANGLES, target_pos = [369,430], render=True, change_goal=False)
    action_dim = env.num_actions
    state_dim  = env.num_observations
else: # gym env
    if ENV == 'CartPole-v0':
        DISCRETE = True
        hidden_dim = 30
    elif ENV == 'Pendulum-v0':
        DISCRETE = False
        hidden_dim = 128
    if DISCRETE:
        env = gym.make(ENV)  # discrete env no normalizedactions
        action_dim = env.action_space.n
    else:
        env = NormalizedActions(gym.make(ENV))
        action_dim = env.action_space.shape[0]
    state_dim  = env.observation_space.shape[0]
    action_range=1.


actor_net = ActorNetwork(state_dim, action_dim, hidden_dim).to(device)
critic_net = CriticNetwork(state_dim, hidden_dim).to(device)
print('Actor Network: ', actor_net)
print('Critic Network: ', critic_net)
actor_optimizer = optim.Adam(actor_net.parameters(), lr=1e-3)
critic_optimizer = optim.Adam(critic_net.parameters(), lr=1e-2)

def train():
    # hyper-parameters
    max_episodes  = 3000
    if ENV ==  'Reacher':
        max_steps   = 20
    elif ENV == 'Pendulum-v0': 
        max_steps   = 150  # Pendulum needs 150 steps per episode to learn well, cannot handle 20
    elif ENV == 'CartPole-v0':
        max_steps   = 1000 # short time step would be too easy for CartPole

    frame_idx   = 0
    running_reward = 10
    episode_rewards = []
    # SavedTuple = namedtuple('SavedSet', ['log_prob', 'state_value'])
    # SavedTuple2 = namedtuple('SavedSet2', ['log_prob', 'state_value', 'next_state_value'])


    for i_episode in range (max_episodes):
        
        if ENV == 'Reacher':
            state = env.reset(SCREEN_SHOT)
        # elif ENV == 'Pendulum':
        else: # gym env
            state =  env.reset()
        episode_reward = 0
        if ON_POLICY:
            rewards=[]
        if not DETERMINISTIC:
            entropies=0
        for step in range (max_steps):
            frame_idx+=1
            if ON_POLICY:            
                action, log_prob, entropy = actor_net.evaluate_action(state)
                # print('state: ', state, 'action: ', action, 'log_prob: ', log_prob)
                state_value = critic_net(state)

                if ENV ==  'Reacher':
                    next_state, reward, done, _ = env.step(action, SPARSE_REWARD, SCREEN_SHOT)
                # elif ENV ==  'Pendulum':
                else: # gym env
                    if DISCRETE:
                        next_state, reward, done, _ = env.step(action[0]) # discrete action only needs a index
                    else:
                        next_state, reward, done, _ = env.step(action)
                    env.render()   
                next_state_value = critic_net(next_state)
                actor_net.saved_entropies.append(entropy)
                if UPDATE == 'Approach0':
                    # this is critical! have to save the values inside the model to keep track of its gradients
                    actor_net.saved_logprobs.append(log_prob)
                    critic_net.saved_values.append(state_value)
                    # SavedSet.append(SavedTuple(log_prob, state_value))
                if UPDATE == 'Approach1':
                    # this is critical! have to save the values inside the model to keep track of its gradients
                    actor_net.saved_logprobs.append(log_prob)
                    critic_net.saved_values.append(state_value)
                    critic_net.saved_nextvalues.append(next_state_value)
                    # SavedSet.append(SavedTuple2(log_prob, state_value, next_state_value))
                if done:
                    reward = -20 if ENV == 'CartPole-v0' else reward
                    break
                rewards.append(reward)
            else: # off-policy update with memory buffer
                pass
                if done:
                    reward = -20 if ENV == 'CartPole-v0' else reward
                    break

            state = next_state
            episode_reward += reward
            running_reward = running_reward * 0.99 + episode_reward * 0.01
            # rewards.append(episode_reward)
            if frame_idx%500==0:
                plot(frame_idx, episode_rewards)


        print('Episode: ', i_episode, '| Episode Reward: ', episode_reward, '| Running Reward: ', running_reward)
        episode_rewards.append(episode_reward)
        if UPDATE == 'Approach0':
            Update0(rewards)
        if UPDATE == 'Approach1':
            Update1(rewards)

def main():
    train()


if __name__ == '__main__':
    main()
        
