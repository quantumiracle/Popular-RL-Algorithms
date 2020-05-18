'''
Soft Actor-Critic version 2
using target Q instead of V net: 2 Q net, 2 target Q net, 1 policy net
add alpha loss compared with version 1
paper: https://arxiv.org/pdf/1812.05905.pdf
'''

import math
import random

import gym
import numpy as np

import torch

torch.multiprocessing.set_start_method('forkserver', force=True)
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

from IPython.display import clear_output
import matplotlib.pyplot as plt
from reacher import Reacher
import argparse

import torch.multiprocessing as mp

from multiprocessing import Process
from multiprocessing.managers import BaseManager


parser = argparse.ArgumentParser(description='Train or test neural net motor controller.')
parser.add_argument('--train', dest='train', action='store_true', default=False)
parser.add_argument('--test', dest='test', action='store_true', default=False)

args = parser.parse_args()


class SharedAdam(optim.Optimizer):
    r"""Implements Adam algorithm.

    It has been proposed in `Adam: A Method for Stochastic Optimization`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(SharedAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SharedAdam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]
                ### ADD
                device = p.device
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
                ### ADD

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad.add_(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg, denom)

        return loss


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
        state, action, reward, next_state, done = map(np.stack,
                                                      zip(*batch))  # stack for each element
        ''' 
        the * serves as unpack: sum(a,b) <=> batch=(a,b), sum(*batch) ;
        zip: a=[1,2], b=[2,3], zip(a,b) => [(1, 2), (2, 3)] ;
        the map serves as mapping the function on each list element: map(square, [2,3]) => [4,9] ;
        np.stack((1,2)) => array([1, 2])
        '''
        return state, action, reward, next_state, done

    def __len__(
            self):  # cannot work in multiprocessing case, len(replay_buffer) is not available in proxy of manager!
        return len(self.buffer)

    def get_length(self):
        return len(self.buffer)


class NormalizedActions(gym.ActionWrapper):
    def _action(self, action):
        low = self.action_space.low
        high = self.action_space.high

        action = low + (action + 1.0) * 0.5 * (high - low)
        action = np.clip(action, low, high)

        return action

    def _reverse_action(self, action):
        low = self.action_space.low
        high = self.action_space.high

        action = 2 * (action - low) / (high - low) - 1
        action = np.clip(action, low, high)

        return action


class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim, init_w=3e-3):
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim)
        self.linear4 = nn.Linear(hidden_dim, 1)
        # weights initialization
        self.linear4.weight.data.uniform_(-init_w, init_w)
        self.linear4.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.linear4(x)
        return x


class SoftQNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3):
        super(SoftQNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, 1)

        self.linear4.weight.data.uniform_(-init_w, init_w)
        self.linear4.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)  # the dim 0 is number of samples
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.linear4(x)
        return x


class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, action_range=1., init_w=3e-3, log_std_min=-20, log_std_max=2):
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

        self.action_range = action_range
        self.num_actions = num_actions

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))

        mean    = (self.mean_linear(x))
        # mean    = F.leaky_relu(self.mean_linear(x))
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std
    
    def evaluate(self, state, epsilon=1e-6):
        '''
        generate sampled action with state as input wrt the policy network;
        '''
        mean, log_std = self.forward(state)
        std = log_std.exp() # no clip in evaluation, clip affects gradients flow
        
        normal = Normal(0, 1)
        z = normal.sample(mean.shape)
        action_0 = torch.tanh(mean + std * z.cuda())  # TanhNormal distribution as actions; reparameterization trick
        action = self.action_range * action_0
        log_prob = Normal(mean, std).log_prob(mean + std * z.cuda()) - torch.log(
            1. - action_0.pow(2) + epsilon) - np.log(self.action_range)
        # both dims of normal.log_prob and -log(1-a**2) are (N,dim_of_action);
        # the Normal.log_prob outputs the same dim of input features instead of 1 dim probability,
        # needs sum up across the features dim to get 1 dim prob; or else use Multivariate Normal.
        log_prob = log_prob.sum(dim=1, keepdim=True)
        return action, log_prob, z, mean, log_std

    def get_action(self, state, deterministic):
        state = torch.FloatTensor(state).unsqueeze(0).cuda()
        # print(state)
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        normal = Normal(0, 1)
        z = normal.sample(mean.shape).cuda()
        action = self.action_range * torch.tanh(mean + std * z)

        action = self.action_range * torch.tanh(mean).detach().cpu().numpy()[0] if deterministic else \
        action.detach().cpu().numpy()[0]
        return action

    def sample_action(self, ):
        a = torch.FloatTensor(self.num_actions).uniform_(-1, 1)
        return self.action_range * a.numpy()

class Alpha(nn.Module):
    ''' nn.Module class of alpha variable, for the usage of parallel on gpus '''
    def __init__(self):
        super(Alpha, self).__init__()
        self.log_alpha=torch.nn.Parameter(torch.zeros(1))  #initialized as [0.]: alpha->[1.]

    def forward(self):
        return self.log_alpha

class SAC_Trainer():
    def __init__(self, replay_buffer, hidden_dim, action_range):
        self.replay_buffer = replay_buffer
        self.action_dim = action_dim

        self.soft_q_net1 = SoftQNetwork(state_dim, action_dim, hidden_dim)
        self.soft_q_net2 = SoftQNetwork(state_dim, action_dim, hidden_dim)
        self.target_soft_q_net1 = SoftQNetwork(state_dim, action_dim, hidden_dim)
        self.target_soft_q_net2 = SoftQNetwork(state_dim, action_dim, hidden_dim)
        self.policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim, action_range)
        self.log_alpha = Alpha()
        print('Soft Q Network (1,2): ', self.soft_q_net1)
        print('Policy Network: ', self.policy_net)

        for target_param, param in zip(self.target_soft_q_net1.parameters(),
                                       self.soft_q_net1.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.target_soft_q_net2.parameters(),
                                       self.soft_q_net2.parameters()):
            target_param.data.copy_(param.data)

        self.soft_q_criterion1 = nn.MSELoss()
        self.soft_q_criterion2 = nn.MSELoss()

        soft_q_lr = 3e-4
        policy_lr = 3e-4
        alpha_lr = 3e-4

        self.soft_q_optimizer1 = SharedAdam(self.soft_q_net1.parameters(), lr=soft_q_lr)
        self.soft_q_optimizer2 = SharedAdam(self.soft_q_net2.parameters(), lr=soft_q_lr)
        self.policy_optimizer = SharedAdam(self.policy_net.parameters(), lr=policy_lr)
        self.alpha_optimizer = SharedAdam(self.log_alpha.parameters(), lr=alpha_lr)

    def to_cuda(self):  # copy to specified gpu
        self.soft_q_net1 = self.soft_q_net1.cuda()
        self.soft_q_net2 = self.soft_q_net2.cuda()
        self.target_soft_q_net1 = self.target_soft_q_net1.cuda()
        self.target_soft_q_net2 = self.target_soft_q_net2.cuda()
        self.policy_net = self.policy_net.cuda()
        self.log_alpha = self.log_alpha.cuda()

    def update(self, batch_size, reward_scale=10., auto_entropy=True, target_entropy=-2, gamma=0.99,
               soft_tau=1e-2):
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)
        # print('sample:', state, action,  reward, done)

        state = torch.FloatTensor(state).cuda()
        next_state = torch.FloatTensor(next_state).cuda()
        action = torch.FloatTensor(action).cuda()
        reward = torch.FloatTensor(reward).unsqueeze(1).cuda()  # reward is single value, unsqueeze() to add one dim to be [reward] at the sample dim;
        done = torch.FloatTensor(np.float32(done)).unsqueeze(1).cuda()

        predicted_q_value1 = self.soft_q_net1(state, action)
        predicted_q_value2 = self.soft_q_net2(state, action)
        new_action, log_prob, z, mean, log_std = self.policy_net.evaluate(state)
        new_next_action, next_log_prob, _, _, _ = self.policy_net.evaluate(next_state)
        reward = reward_scale * (reward - reward.mean(dim=0)) / (reward.std(
            dim=0) + 1e-6)  # normalize with batch mean and std; plus a small number to prevent numerical problem

        # Updating alpha wrt entropy
        # alpha = 0.0
        # trade-off between exploration (max entropy) and exploitation (max Q)
        if auto_entropy is True:
            alpha_loss = -(self.log_alpha() * (log_prob - 1.0 * self.action_dim).detach()).mean()  # self.log_alpha as forward function to get value
            # print('alpha loss: ',alpha_loss)
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha().exp()
        else:
            self.alpha = 1.
            alpha_loss = 0

        # print(self.alpha)
        # Training Q Function
        target_q_min = torch.min(self.target_soft_q_net1(next_state, new_next_action),
                                 self.target_soft_q_net2(next_state,
                                                         new_next_action)) - self.alpha * next_log_prob
        target_q_value = reward + (1 - done) * gamma * target_q_min  # if done==1, only reward
        q_value_loss1 = self.soft_q_criterion1(predicted_q_value1,
                                               target_q_value.detach())  # detach: no gradients for the variable
        q_value_loss2 = self.soft_q_criterion2(predicted_q_value2, target_q_value.detach())

        self.soft_q_optimizer1.zero_grad()
        q_value_loss1.backward()
        self.soft_q_optimizer1.step()
        self.soft_q_optimizer2.zero_grad()
        q_value_loss2.backward()
        self.soft_q_optimizer2.step()

        # Training Policy Function
        predicted_new_q_value = torch.min(self.soft_q_net1(state, new_action),
                                          self.soft_q_net2(state, new_action))
        policy_loss = (self.alpha * log_prob - predicted_new_q_value).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # print('q loss: ', q_value_loss1, q_value_loss2)
        # print('policy loss: ', policy_loss )

        # Soft update the target value net
        for target_param, param in zip(self.target_soft_q_net1.parameters(),
                                       self.soft_q_net1.parameters()):
            target_param.data.copy_(  # copy data value into target parameters
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )
        for target_param, param in zip(self.target_soft_q_net2.parameters(),
                                       self.soft_q_net2.parameters()):
            target_param.data.copy_(  # copy data value into target parameters
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )
        return predicted_new_q_value.mean()

    def save_model(self, path):
        torch.save(self.soft_q_net1.state_dict(),
                   path + '_q1')  # have to specify different path name here!
        torch.save(self.soft_q_net2.state_dict(), path + '_q2')
        torch.save(self.policy_net.state_dict(), path + '_policy')

    def load_model(self, path):
        self.soft_q_net1.load_state_dict(torch.load(path + '_q1', map_location='cuda:0'))  # map model on single gpu for testing
        self.soft_q_net2.load_state_dict(torch.load(path + '_q2', map_location='cuda:0'))
        self.policy_net.load_state_dict(torch.load(path + '_policy', map_location='cuda:0'))

        self.soft_q_net1.eval()
        self.soft_q_net2.eval()
        self.policy_net.eval()


def worker(id, sac_trainer, ENV, rewards_queue, replay_buffer, max_episodes, max_steps, batch_size,
           explore_steps, \
           update_itr, action_itr, AUTO_ENTROPY, DETERMINISTIC, hidden_dim, model_path):
    '''
    the function for sampling with multi-processing
    '''

    with torch.cuda.device(id % torch.cuda.device_count()):
        sac_trainer.to_cuda()

        print(sac_trainer, replay_buffer)  # sac_tainer are not the same, but all networks and optimizers in it are the same; replay  buffer is the same one.
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
            action_dim = env.num_actions
            state_dim  = env.num_observations

        elif ENV == 'Pendulum':
            env = NormalizedActions(gym.make("Pendulum-v0"))
            action_dim = env.action_space.shape[0]
            state_dim  = env.observation_space.shape[0]
            action_range=1.
        
        frame_idx=0
        rewards=[]
        # training loop
        for eps in range(max_episodes):
            episode_reward = 0
            if ENV == 'Reacher':
                state = env.reset(SCREEN_SHOT)
            elif ENV == 'Pendulum':
                state =  env.reset()
            
            for step in range(max_steps):
                if frame_idx > explore_steps:
                    action = sac_trainer.policy_net.get_action(state, deterministic = DETERMINISTIC)
                else:
                    action = sac_trainer.policy_net.sample_action()
        
                try:
                    if ENV ==  'Reacher':
                        next_state, reward, done, _ = env.step(action, SPARSE_REWARD, SCREEN_SHOT)
                    elif ENV ==  'Pendulum':
                        next_state, reward, done, _ = env.step(action)
                        env.render() 
                except KeyboardInterrupt:
                    print('Finished')
                    sac_trainer.save_model(model_path)

                replay_buffer.push(state, action, reward, next_state, done)

                state = next_state
                episode_reward += reward
                frame_idx += 1

                # if len(replay_buffer) > batch_size:
                if replay_buffer.get_length() > batch_size:
                    for i in range(update_itr):
                        _ = sac_trainer.update(batch_size, reward_scale=10., auto_entropy=AUTO_ENTROPY,
                                                target_entropy=-1. * action_dim)

                if eps % 10 == 0 and eps > 0:
                    # plot(rewards, id)
                    sac_trainer.save_model(model_path)

                if done:
                    break
            print('Worker: ', id, '| Episode: ', eps, '| Episode Reward: ', episode_reward)
            # if len(rewards) == 0:
            #     rewards.append(episode_reward)
            # else:
            #     rewards.append(rewards[-1] * 0.9 + episode_reward * 0.1)
            rewards_queue.put(episode_reward)

        sac_trainer.save_model(model_path)


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
    plt.figure(figsize=(20, 5))
    plt.plot(rewards)
    plt.savefig('sac_v2_multi.png')
    # plt.show()
    plt.clf()


if __name__ == '__main__':

    replay_buffer_size = 1e6

    # the replay buffer is a class, have to use torch manager to make it a proxy for sharing across processes
    BaseManager.register('ReplayBuffer', ReplayBuffer)
    manager = BaseManager()
    manager.start()
    replay_buffer = manager.ReplayBuffer(
        replay_buffer_size)  # share the replay buffer through manager

# choose env
    ENV = ['Pendulum', 'Reacher'][0]
    if ENV == 'Reacher':
        NUM_JOINTS=2
        LINK_LENGTH=[200, 140]
        SCREEN_SIZE=1000
        SPARSE_REWARD=False
        SCREEN_SHOT=False
        action_range = 10.0

        env=Reacher(screen_size=SCREEN_SIZE, num_joints=NUM_JOINTS, link_lengths = LINK_LENGTH, \
        ini_joint_angles=INI_JOING_ANGLES, target_pos = [369,430], render=True, change_goal=False)
        action_dim = env.num_actions
        state_dim  = env.num_observations

    elif ENV == 'Pendulum':
        env = NormalizedActions(gym.make("Pendulum-v0"))
        action_dim = env.action_space.shape[0]
        state_dim  = env.observation_space.shape[0]
        action_range=1.

    # hyper-parameters for RL training, no need for sharing across processes
    max_episodes = 1000
    max_steps   = 20 if ENV ==  'Reacher' else 150  # Pendulum needs 150 steps per episode to learn well, cannot handle 20
    explore_steps = 0  # for random action sampling in the beginning of training
    batch_size = 640
    update_itr = 1
    action_itr = 3
    AUTO_ENTROPY = True
    DETERMINISTIC = False
    hidden_dim = 512
    model_path = './sac_model/sac_v2_multiprocess_multi'

    sac_trainer = SAC_Trainer(replay_buffer, hidden_dim=hidden_dim, action_range=action_range)

    if args.train:

        # share the global parameters in multiprocessing
        sac_trainer.soft_q_net1.share_memory()
        sac_trainer.soft_q_net2.share_memory()
        sac_trainer.target_soft_q_net1.share_memory()
        sac_trainer.target_soft_q_net2.share_memory()
        sac_trainer.policy_net.share_memory()
        ShareParameters(sac_trainer.soft_q_optimizer1)
        ShareParameters(sac_trainer.soft_q_optimizer2)
        ShareParameters(sac_trainer.policy_optimizer)
        ShareParameters(sac_trainer.alpha_optimizer)

        rewards_queue = mp.Queue()  # used for get rewards from all processes and plot the curve

        num_workers = 2  # or: mp.cpu_count()
        processes = []
        rewards = [0]

        for i in range(num_workers):
            process = Process(target=worker, args=(
            i, sac_trainer, ENV, rewards_queue, replay_buffer, max_episodes, max_steps, \
            batch_size, explore_steps, update_itr, action_itr, AUTO_ENTROPY, DETERMINISTIC,
            hidden_dim, model_path))  # the args contain shared and not shared
            process.daemon = True  # all processes closed when the main stops
            processes.append(process)

        [p.start() for p in processes]
        while True:  # keep geting the episode reward from the queue
            r = rewards_queue.get()
            if r is not None:
                rewards.append(0.9 * rewards[-1] + 0.1 * r)  # moving average of episode rewards
            else:
                break

            if len(rewards) % 20 == 0 and len(rewards) > 0:
                plot(rewards)

        [p.join() for p in processes]  # finished at the same time

        sac_trainer.save_model(model_path)

    if args.test:
        # single process for testing
        env = L2RunEnv(visualize=True)  # L2M2019Env
        sac_trainer.load_model(model_path)
        sac_trainer.to_cuda()  # from cpu to cuda

        for eps in range(10):
            if ENV == 'Reacher':
                state = env.reset(SCREEN_SHOT)
            elif ENV == 'Pendulum':
                state =  env.reset()
            episode_reward = 0

            for step in range(max_steps):
                action = sac_trainer.policy_net.get_action(state, deterministic = DETERMINISTIC)
                if ENV ==  'Reacher':
                    next_state, reward, done, _ = env.step(action, SPARSE_REWARD, SCREEN_SHOT)
                elif ENV ==  'Pendulum':
                    next_state, reward, done, _ = env.step(action)
                    env.render()   

            print('Episode: ', eps, '| Episode Reward: ', episode_reward)
