
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import time
import random, numpy, argparse, logging, os
from collections import namedtuple
import numpy as np
import datetime, math
import gym

# Hyper Parameters
MAX_EPI=10000
MAX_STEP = 10000
SAVE_INTERVAL = 20
TARGET_UPDATE_INTERVAL = 20

BATCH_SIZE = 128
REPLAY_BUFFER_SIZE = 100000
REPLAY_START_SIZE = 2000

GAMMA = 0.95
EPSILON = 0.05  # if not using epsilon scheduler, use a constant
EPSILON_START = 1.
EPSILON_END = 0.05
EPSILON_DECAY = 10000
LR = 1e-4                  # learning rate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EpsilonScheduler():
    def __init__(self, eps_start, eps_final, eps_decay):
        """A scheduler for epsilon-greedy strategy.

        :param eps_start: starting value of epsilon, default 1. as purely random policy 
        :type eps_start: float
        :param eps_final: final value of epsilon
        :type eps_final: float
        :param eps_decay: number of timesteps from eps_start to eps_final
        :type eps_decay: int
        """
        self.eps_start = eps_start
        self.eps_final = eps_final
        self.eps_decay = eps_decay
        self.epsilon = self.eps_start
        self.ini_frame_idx = 0
        self.current_frame_idx = 0

    def reset(self, ):
        """ Reset the scheduler """
        self.ini_frame_idx = self.current_frame_idx

    def step(self, frame_idx):
        self.current_frame_idx = frame_idx
        delta_frame_idx = self.current_frame_idx - self.ini_frame_idx
        self.epsilon = self.eps_final + (self.eps_start - self.eps_final) * math.exp(-1. * delta_frame_idx / self.eps_decay)
    
    def get_epsilon(self):
        return self.epsilon


class QNetwork(nn.Module):
    def __init__(self, act_shape, obs_shape, hidden_units=64):
        super(QNetwork, self).__init__()
        in_dim = obs_shape[0]
        out_dim = act_shape

        self.linear = nn.Sequential(
            nn.Linear(in_dim, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, out_dim)
        )

    def forward(self, x):
        o = self.linear(x)
        return o

class QNetworkCNN(nn.Module):
    def __init__(self, num_actions, in_shape, out_channels=8, kernel_size=5, stride=1, hidden_units=256):
        super(QNetworkCNN, self).__init__()

        self.in_shape = in_shape
        in_channels = in_shape[0]
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, int(out_channels/2), kernel_size, stride),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size, stride=2),
            nn.Conv2d(int(out_channels/2), int(out_channels), kernel_size, stride),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size, stride=2)
        )
        self.conv.apply(self.init_weights)

        self.linear = nn.Sequential(
            nn.Linear(self.size_after_conv(), hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, num_actions)
        )

        self.linear.apply(self.init_weights)

    def init_weights(self, m):
        if type(m) == nn.Conv2d or type(m) == nn.Linear:
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)

    def size_after_conv(self,):
        x = torch.rand(1, *self.in_shape)
        o = self.conv(x)
        size=1
        for i in o.shape[1:]:
            size*=i
        return int(size)

    def forward(self, x):
        x = self.conv(x)
        o = self.linear(x.view(x.size(0), -1))
        return o

transition = namedtuple('transition', 'state, next_state, action, reward, is_terminal')
class replay_buffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.location = 0
        self.buffer = []

    def add(self, samples):
        # Append when the buffer is not full but overwrite when the buffer is full
        wrap_tensor = lambda x: torch.tensor([x])
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(transition(*map(wrap_tensor, samples)))
        else:
            self.buffer[self.location] = transition(*map(wrap_tensor, samples))

        # Increment the buffer location
        self.location = (self.location + 1) % self.buffer_size

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

class DQN(object):
    def __init__(self, env):
        self.action_shape = env.action_space.n
        self.obs_shape = env.observation_space.shape
        self.eval_net, self.target_net = QNetwork(self.action_shape, self.obs_shape).to(device), QNetwork(self.action_shape, self.obs_shape).to(device)
        self.learn_step_counter = 0                                     # for target updating
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()
        self.epsilon_scheduler = EpsilonScheduler(EPSILON_START, EPSILON_END, EPSILON_DECAY)
        self.updates = 0

    def choose_action(self, x):
        # x = Variable(torch.unsqueeze(torch.FloatTensor(x), 0)).to(device)
        x = torch.unsqueeze(torch.FloatTensor(x), 0).to(device)
        # input only one sample
        # if np.random.uniform() > EPSILON:   # greedy
        epsilon = self.epsilon_scheduler.get_epsilon()
        if np.random.uniform() > epsilon:   # greedy
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.cpu().numpy()[0]     # return the argmax
            # print(action)
        else:   # random
            action = np.random.randint(0, self.action_shape)
        return action

    def learn(self, sample,):
        # Batch is a list of namedtuple's, the following operation returns samples grouped by keys
        batch_samples = transition(*zip(*sample))

        # states, next_states are of tensor (BATCH_SIZE, in_channel, 10, 10) - inline with pytorch NCHW format
        # actions, rewards, is_terminal are of tensor (BATCH_SIZE, 1)
        states = torch.cat(batch_samples.state).float().to(device)
        next_states = torch.cat(batch_samples.next_state).float().to(device)
        actions = torch.cat(batch_samples.action).to(device)
        rewards = torch.cat(batch_samples.reward).float().to(device)
        is_terminal = torch.cat(batch_samples.is_terminal).to(device)
        # Obtain a batch of Q(S_t, A_t) and compute the forward pass.
        # Note: policy_network output Q-values for all the actions of a state, but all we need is the A_t taken at time t
        # in state S_t.  Thus we gather along the columns and get the Q-values corresponds to S_t, A_t.
        # Q_s_a is of size (BATCH_SIZE, 1).
        Q = self.eval_net(states) 
        Q_s_a=Q.gather(1, actions)

        # Obtain max_{a} Q(S_{t+1}, a) of any non-terminal state S_{t+1}.  If S_{t+1} is terminal, Q(S_{t+1}, A_{t+1}) = 0.
        # Note: each row of the network's output corresponds to the actions of S_{t+1}.  max(1)[0] gives the max action
        # values in each row (since this a batch).  The detach() detaches the target net's tensor from computation graph so
        # to prevent the computation of its gradient automatically.  Q_s_prime_a_prime is of size (BATCH_SIZE, 1).

        # Get the indices of next_states that are not terminal
        none_terminal_next_state_index = torch.tensor([i for i, is_term in enumerate(is_terminal) if is_term == 0], dtype=torch.int64, device=device)
        # Select the indices of each row
        none_terminal_next_states = next_states.index_select(0, none_terminal_next_state_index)

        Q_s_prime_a_prime = torch.zeros(len(sample), 1, device=device)
        if len(none_terminal_next_states) != 0:
            Q_s_prime_a_prime[none_terminal_next_state_index] = self.target_net(none_terminal_next_states).detach().max(1)[0].unsqueeze(1)

        # Q_s_prime_a_prime = self.target_net(next_states).detach().max(1, keepdim=True)[0]  # this one is simpler regardless of terminal state
        Q_s_prime_a_prime = (Q_s_prime_a_prime-Q_s_prime_a_prime.mean())/ (Q_s_prime_a_prime.std() + 1e-5)  # normalization
        
        # Compute the target
        target = rewards + GAMMA * Q_s_prime_a_prime

        # Update with loss
        # loss = self.loss_func(target.detach(), Q_s_a)
        loss = f.smooth_l1_loss(target.detach(), Q_s_a)
        # Zero gradients, backprop, update the weights of policy_net
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.updates += 1
        if self.updates % TARGET_UPDATE_INTERVAL == 0:
            self.update_target()

        return loss.item()

    def save_model(self, model_path=None):
        torch.save(self.eval_net.state_dict(), 'model/dqn')

    def update_target(self, ):
        """
        Update the target model when necessary.
        """
        self.target_net.load_state_dict(self.eval_net.state_dict())
    
def rollout(env, model):
    r_buffer = replay_buffer(REPLAY_BUFFER_SIZE)
    log = []
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    print('\nCollecting experience...')
    total_step = 0
    for epi in range(MAX_EPI):
        s=env.reset()
        epi_r = 0
        epi_loss = 0
        for step in range(MAX_STEP):
            # env.render()
            total_step += 1
            a = model.choose_action(s)
            s_, r, done, info = env.step(a)
            # r_buffer.add(torch.tensor([s]), torch.tensor([s_]), torch.tensor([[a]]), torch.tensor([[r]], dtype=torch.float), torch.tensor([[done]]))
            r_buffer.add([s,s_,[a],[r],[done]])
            model.epsilon_scheduler.step(total_step)
            epi_r += r
            if total_step > REPLAY_START_SIZE and len(r_buffer.buffer) >= BATCH_SIZE:
                sample = r_buffer.sample(BATCH_SIZE)
                loss = model.learn(sample)
                epi_loss += loss
            if done:
                break
            s = s_
        print('Ep: ', epi, '| Ep_r: ', epi_r, '| Steps: ', step, f'| Ep_Loss: {epi_loss:.4f}', )
        log.append([epi, epi_r, step])
        # if epi % SAVE_INTERVAL == 0:
            # model.save_model()
            # np.save('log/'+timestamp, log)

if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    print(env.observation_space, env.action_space)
    model = DQN(env)
    rollout(env, model)