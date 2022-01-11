### QMIX algorithm 
# paper: QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning
# reference: https://github.com/AI4Finance-Foundation/ElegantRL/blob/e980158e89cdc3c80be9c0770790a84dc6db8efd/elegantrl/agents/AgentQMix.py

from numpy.core.function_base import _logspace_dispatcher
from pettingzoo.butterfly import cooperative_pong_v3  # cannot use ram
from pettingzoo.atari import entombed_cooperative_v2
import numpy as np
from common.wrappers import Dict2TupleWrapper
import supersuit
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import random
from os import path
import pickle
import argparse

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

def wrap_env(env, obs_type='ram'):
    env = env.parallel_env(obs_type=obs_type)
    env_agents = env.unwrapped.agents
    if obs_type == 'rgb_image':
        env = supersuit.max_observation_v0(env, 2)  # as per openai baseline's MaxAndSKip wrapper, maxes over the last 2 frames to deal with frame flickering
        env = supersuit.sticky_actions_v0(env, repeat_action_probability=0.25) # repeat_action_probability is set to 0.25 to introduce non-determinism to the system
        env = supersuit.frame_skip_v0(env, 4) # skip frames for faster processing and less control to be compatable with gym, use frame_skip(env, (2,5))
        env = supersuit.resize_v0(env, 84, 84) # downscale observation for faster processing
        env = supersuit.frame_stack_v1(env, 4) # allow agent to see everything on the screen despite Atari's flickering screen problem
    else:
        env = supersuit.frame_skip_v0(env, 4)  # RAM version also need frame skip, essential for boxing-v1, etc
            
    # normalize the observation of Atari for both image or RAM 
    env = supersuit.dtype_v0(env, 'float32') # need to transform uint8 to float first for normalizing observation: https://github.com/PettingZoo-Team/SuperSuit
    env = supersuit.normalize_obs_v0(env, env_min=0, env_max=1) # normalize the observation to (0,1)
    
    env.observation_space = list(env.observation_spaces.values())[0]
    env.action_space = list(env.action_spaces.values())[0]
    env.agents = env_agents
    env = Dict2TupleWrapper(env) 

    return env

class ReplayBufferGRU:
    """ 
    Replay buffer for agent with GRU network additionally storing previous action, 
    initial input hidden state and output hidden state of GRU.
    And each sample contains the whole episode instead of a single step.
    'hidden_in' and 'hidden_out' are only the initial hidden state for each episode, for GRU initialization.

    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, hidden_in, hidden_out, state, action, last_action, reward, next_state):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (
            hidden_in, hidden_out, state, action, last_action, reward, next_state)
        self.position = int((self.position + 1) %
                            self.capacity)  # as a ring buffer

    def sample(self, batch_size):
        s_lst, a_lst, la_lst, r_lst, ns_lst, hi_lst, ho_lst = [], [], [], [], [], [], []
        batch = random.sample(self.buffer, batch_size)
        min_seq_len = float('inf')
        for sample in batch:
            h_in, h_out, state, action, last_action, reward, next_state = sample
            min_seq_len = min(len(state), min_seq_len)
            hi_lst.append(h_in)  # h_in: (1, batch_size=1, n_agents, hidden_size)
            ho_lst.append(h_out)
        hi_lst = torch.cat(hi_lst, dim=-3).detach()  # cat along the batch dim
        ho_lst = torch.cat(ho_lst, dim=-3).detach()

        # strip sequence length
        for sample in batch:
            h_in, h_out, state, action, last_action, reward, next_state = sample
            sample_len = len(state)
            start_idx = int((sample_len - min_seq_len)/2)
            end_idx = start_idx+min_seq_len
            s_lst.append(state[start_idx:end_idx])
            a_lst.append(action[start_idx:end_idx])
            la_lst.append(last_action[start_idx:end_idx])
            r_lst.append(reward[start_idx:end_idx])
            ns_lst.append(next_state[start_idx:end_idx])
        # print("s_lst.shape: {}".format(np.array(s_lst).shape))
        # print("a_lst.shape: {}".format(np.array(a_lst).shape))
        # print("la_lst.shape: {}".format(np.array(la_lst).shape))
        # print("r_lst.shape: {}".format(np.array(r_lst).shape))
        # print("ns_lst.shape: {}".format(np.array(ns_lst).shape))

        return hi_lst, ho_lst, s_lst, a_lst, la_lst, r_lst, ns_lst

    def __len__(
            self):  # cannot work in multiprocessing case, len(replay_buffer) is not available in proxy of manager!
        return len(self.buffer)

    def get_length(self):
        return len(self.buffer)

    def dump_buffer(self):

        # Saving the objects:
        with open(self.save2file, 'wb') as f:  # Python 3: open(..., 'wb')
            pickle.dump([self.buffer, self.position], f)

class RNNAgent(nn.Module):
    '''
    @brief:
        evaluate Q value given a state and the action
    '''

    def __init__(self, num_inputs, action_shape, num_actions, hidden_size):
        super(RNNAgent, self).__init__()

        self.num_inputs = num_inputs
        self.action_shape = action_shape
        self.num_actions = num_actions

        self.linear1 = nn.Linear(num_inputs+action_shape*num_actions, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.rnn = nn.GRU(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, action_shape*num_actions)

    def forward(self, state, action, hidden_in):
        '''
        @params:
            state: [#batch, #sequence, #agent, #n_feature]
            action: [#batch, #sequence, #agent, action_shape]
        @return:
            qs: [#batch, #sequence, #agent, action_shape, num_actions]
        '''
        #  to [#sequence, #batch, #agent, #n_feature]
        bs, seq_len, n_agents, _= state.shape
        state = state.permute(1, 0, 2, 3)
        action = action.permute(1, 0, 2, 3)
        action = F.one_hot(action, num_classes=self.num_actions)
        action = action.view(seq_len, bs, n_agents, -1) # [#batch, #sequence, #agent, action_shape*num_actions]

        x = torch.cat([state, action], -1)  # the dim 0 is number of samples
        x = x.view(seq_len, bs*n_agents, -1) # change x to [#sequence, #batch*#agent, -1] to meet rnn's input requirement
        hidden_in = hidden_in.view(1, bs*n_agents, -1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x,  hidden = self.rnn(x, hidden_in)
        x = F.relu(self.linear3(x))
        x = self.linear4(x) # [#sequence, #batch, #agents, #action_shape*#actions]
        # [#sequence, #batch, #agent, #head * #action]
        x = x.view(seq_len, bs, n_agents, self.action_shape, self.num_actions)
        hidden = hidden.view(1, bs, n_agents, -1)
        # categorical over the discretized actions
        qs = F.softmax(x, dim=-1)
        qs = qs.permute(1, 0, 2, 3, 4)  # permute back [#batch, #sequence, #agents, #action_shape, #actions]

        return qs, hidden

    def get_action(self, state, last_action, hidden_in, deterministic=False):
        '''
        @brief:
            for each distributed agent, generate action for one step given input data
        @params:
            state: [n_agents, n_feature]
            last_action: [n_agents, action_shape]
        '''
        state = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(device) # add #sequence and #batch: [[#batch, #sequence, n_agents, n_feature]] 
        last_action = torch.LongTensor(
            last_action).unsqueeze(0).unsqueeze(0).to(device)  # add #sequence and #batch: [#batch, #sequence, n_agents, action_shape]
        hidden_in = hidden_in.unsqueeze(1) # add #batch: [#batch, n_agents, hidden_dim]
        agent_outs, hidden_out = self.forward(state, last_action, hidden_in)  # agents_out: [#batch, #sequence, n_agents, action_shape, action_dim]; hidden_out same as hidden_in
        dist = Categorical(agent_outs)

        if deterministic:
            action = np.argmax(agent_outs.detach().cpu().numpy(), axis=-1)
        else:
            action = dist.sample().squeeze(0).squeeze(0).detach().cpu().numpy()  # squeeze the added #batch and #sequence dimension
        return action, hidden_out  # [n_agents, action_shape]

class QMix(nn.Module):
    def __init__(self, state_dim, n_agents, action_shape, embed_dim=64, hypernet_embed=128, abs=True):
        """
        Critic network class for Qmix. Outputs centralized value function predictions given independent q value.
        :param args: (argparse) arguments containing relevant model information.
        """
        super(QMix, self).__init__()

        self.n_agents = n_agents
        self.state_dim = state_dim*n_agents*action_shape # #features*n_agents
        self.action_shape = action_shape

        self.embed_dim = embed_dim
        self.hypernet_embed = hypernet_embed
        self.abs = abs

        self.hyper_w_1 = nn.Sequential(nn.Linear(self.state_dim, self.hypernet_embed),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(self.hypernet_embed, self.action_shape * self.embed_dim * self.n_agents))
        self.hyper_w_final = nn.Sequential(nn.Linear(self.state_dim, self.hypernet_embed),
                                            nn.ReLU(inplace=True),
                                           nn.Linear(self.hypernet_embed, self.embed_dim))

        # State dependent bias for hidden layer
        self.hyper_b_1 = nn.Linear(
            self.state_dim, self.embed_dim)

        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                               nn.ReLU(inplace=True),
                               nn.Linear(self.embed_dim, 1))

    def forward(self, agent_qs, states):
        """
        Compute actions from the given inputs.
        @params:
            agent_qs: [#batch, #sequence, #agent, #action_shape]
            states: [#batch, #sequence, #agent, #features*action_shape]
        :param agent_qs: q value inputs into network [batch_size, #agent, action_shape]
        :param states: state observation.
        :return q_tot: (torch.Tensor) return q-total .
        """
        bs = agent_qs.size(0)
        states = states.reshape(-1, self.state_dim)  # [#batch*#sequence, action_shape*#features*#agent]
        agent_qs = agent_qs.reshape(-1, 1, self.n_agents*self.action_shape)  # [#batch*#sequence, 1, #agent*#action_shape]
        # First layer
        w1 = self.hyper_w_1(states).abs() if self.abs else self.hyper_w_1(states)  # [#batch*#sequence, action_shape*self.embed_dim*#agent]
        b1 = self.hyper_b_1(states)  # [#batch*#sequence, self.embed_dim]
        w1 = w1.view(-1, self.n_agents*self.action_shape, self.embed_dim)  # [#batch*#sequence, #agent*action_shape, self.embed_dim]
        b1 = b1.view(-1, 1, self.embed_dim)   # [#batch*#sequence, 1, self.embed_dim]
        hidden = F.elu(torch.bmm(agent_qs, w1) + b1)  # [#batch*#sequence, 1, self.embed_dim]

        # Second layer
        w_final = self.hyper_w_final(states).abs() if self.abs else self.hyper_w_final(states)  # [#batch*#sequence, self.embed_dim]
        w_final = w_final.view(-1, self.embed_dim, 1)  # [#batch*#sequence, self.embed_dim, 1]
        # State-dependent bias
        v = self.V(states).view(-1, 1, 1)  # [#batch*#sequence, 1, 1]
        # Compute final output
        y = torch.bmm(hidden, w_final) + v  
        # Reshape and return
        q_tot = y.view(bs, -1, 1) # [#batch, #sequence, 1]
        return q_tot

    def k(self, states):
        bs = states.size(0)
        w1 = torch.abs(self.hyper_w_1(states))
        w_final = torch.abs(self.hyper_w_final(states))
        w1 = w1.view(-1, self.n_agents, self.embed_dim*self.action_shape)
        w_final = w_final.view(-1, self.embed_dim*self.action_shape, 1)
        k = torch.bmm(w1, w_final).view(bs, -1, self.n_agents)
        k = k / torch.sum(k, dim=2, keepdim=True)
        return k

    def b(self, states):
        bs = states.size(0)
        w_final = torch.abs(self.hyper_w_final(states))
        w_final = w_final.view(-1, self.embed_dim*self.action_shape, 1)
        b1 = self.hyper_b_1(states)
        b1 = b1.view(-1, 1, self.embed_dim*self.action_shape)
        v = self.V(states).view(-1, 1, 1)
        b = torch.bmm(b1, w_final) + v
        return b


class QMix_Trainer():
    def __init__(self, replay_buffer, n_agents, state_dim, action_shape, action_dim, hidden_dim, hypernet_dim, target_update_interval, lr=0.001, logger=None):
        self.replay_buffer = replay_buffer

        self.action_dim = action_dim
        self.action_shape = action_shape
        self.n_agents = n_agents
        self.target_update_interval = target_update_interval
        self.agent = RNNAgent(state_dim, action_shape,
                              action_dim, hidden_dim).to(device)
        self.target_agent = RNNAgent(
            state_dim, action_shape, action_dim, hidden_dim).to(device)
        
        self.mixer = QMix(state_dim, n_agents, action_shape,
                          hidden_dim, hypernet_dim).to(device)
        self.target_mixer = QMix(state_dim, n_agents, action_shape,
                          hidden_dim, hypernet_dim).to(device)
        
        self._update_targets()
        self.update_cnt = 0
        
        self.criterion = nn.MSELoss()

        self.optimizer = optim.Adam(
            list(self.agent.parameters())+list(self.mixer.parameters()), lr=lr)

    def sample_action(self):
        probs = torch.FloatTensor(
            np.ones(self.action_dim)/self.action_dim).to(device)
        dist = Categorical(probs)
        action = dist.sample((self.n_agents, self.action_shape))

        return action.type(torch.FloatTensor).numpy()

    def get_action(self, state, last_action, hidden_in, deterministic=False):
        '''
        @return:
            action: w/ shape [#active_as]
        '''

        action, hidden_out = self.agent.get_action(state, last_action, hidden_in, deterministic=deterministic)

        return action, hidden_out

    def push_replay_buffer(self, ini_hidden_in, ini_hidden_out, episode_state, episode_action, episode_last_action,
                           episode_reward, episode_next_state):
        '''
        @brief: push arguments into replay buffer
        '''
        self.replay_buffer.push(ini_hidden_in, ini_hidden_out, episode_state, episode_action, episode_last_action,
                                episode_reward, episode_next_state)

    def update(self, batch_size):
        hidden_in, hidden_out, state, action, last_action, reward, next_state = self.replay_buffer.sample(
            batch_size)

        state = torch.FloatTensor(state).to(device) # [#batch, sequence, #agents, #features*action_shape]
        next_state = torch.FloatTensor(next_state).to(device)
        action = torch.LongTensor(action).to(device) # [#batch, sequence, #agents, #action_shape]
        last_action = torch.LongTensor(last_action).to(device)
        reward = torch.FloatTensor(reward).unsqueeze(-1).to(device) # reward is scalar, add 1 dim to be [reward] at the same dim

        agent_outs, _ = self.agent(state, last_action, hidden_in) # [#batch, #sequence, #agent, action_shape, num_actions]
        
        chosen_action_qvals = torch.gather(  # [#batch, #sequence, #agent, action_shape]
            agent_outs, dim=-1, index=action.unsqueeze(-1)).squeeze(-1)

        qtot = self.mixer(chosen_action_qvals, state) # [#batch, #sequence, 1]

        # target q
        target_agent_outs, _ = self.target_agent(next_state, action, hidden_out)
        target_max_qvals = target_agent_outs.max(dim=-1, keepdim=True)[0] # [#batch, #sequence, #agents, action_shape]
        target_qtot = self.target_mixer(target_max_qvals, next_state)
        
        reward = reward[:, :, 0]  # reward is the same for agents, so take one
        targets = self._build_td_lambda_targets(reward, target_qtot)

        loss = self.criterion(qtot, targets.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_cnt += 1
        if self.update_cnt % self.target_update_interval == 0:
            self._update_targets()

        return loss.item()

    def _build_td_lambda_targets(self, rewards, target_qs, gamma=0.99, td_lambda=0.6):
        '''
        @params:
            rewards: [#batch, #sequence, 1]
            target_qs: [#batch, #sequence, 1]
        '''
        ret = target_qs.new_zeros(*target_qs.shape)
        ret[:, -1] = target_qs[:, -1]
        # backwards recursive update of the "forward view"
        for t in range(ret.shape[1] - 2, -1, -1):
            ret[:, t] = td_lambda * gamma * ret[:, t+1] + (rewards[:, t] + (1 - td_lambda) * gamma * target_qs[:, t+1])
        return ret

    def _update_targets(self):
        for target_param, param in zip(self.target_mixer.parameters(), self.mixer.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.target_agent.parameters(), self.agent.parameters()):
            target_param.data.copy_(param.data)

    def save_model(self, path):
        torch.save(self.agent.state_dict(), path+'_agent')
        torch.save(self.mixer.state_dict(), path+'_mixer')

    def load_model(self, path):
        self.agent.load_state_dict(torch.load(path+'_agent'))
        self.mixer.load_state_dict(torch.load(path+'_mixer'))

        self.agent.eval()
        self.mixer.eval()


if __name__ == '__main__':
    replay_buffer_size = 1e4
    hidden_dim = 64
    hypernet_dim = 128
    max_steps = 1000
    max_episodes = 1000
    update_iter  = 1
    batch_size = 2
    save_interval = 10
    target_update_interval = 10
    model_path = 'model/qmix'
 
    env = entombed_cooperative_v2  # this is not a valid env, reward seems to be zero-sum; for QMIX we need same reward for all agents
    env = wrap_env(env, obs_type='ram')
    print(env.action_space, env.observation_space)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    action_shape = 1 # compared with the orginal implementation, we allow multi-categorical with action_shape digits, normally for single category action_shape=1
    n_agents = len(env.agents)
    print(state_dim, action_dim, n_agents)

    replay_buffer = ReplayBufferGRU(replay_buffer_size)
    learner = QMix_Trainer(replay_buffer, n_agents, state_dim, action_shape, action_dim, hidden_dim, hypernet_dim, target_update_interval)

    loss = None

    for epi in range(max_episodes):
        # initialize
        hidden_out = torch.zeros([1, n_agents, hidden_dim], dtype=torch.float).to(device)
        last_action = learner.sample_action()
        episode_state = []
        episode_action = []
        episode_last_action = []
        episode_reward = []
        episode_next_state = []

        state = env.reset()

        for step in range(max_steps):
            hidden_in = hidden_out
            action, hidden_out = learner.get_action(
                state, last_action, hidden_in)

            # take next step
            next_state, reward, done, info = env.step(action.reshape(-1)) # [#n_agents, action_shape] to [#n_agents]

            if step == 0:   
                ini_hidden_in = hidden_in
                ini_hidden_out = hidden_out
            episode_state.append(state)
            episode_action.append(action)

            episode_last_action.append(last_action)
            episode_reward.append(reward)
            episode_next_state.append(next_state)

            state = next_state
            last_action = action

            # break the episode
            if np.any(done):
                break

        # update SAC
        if args.train:
            learner.push_replay_buffer(ini_hidden_in, ini_hidden_out, episode_state, episode_action, episode_last_action,
                                episode_reward, episode_next_state)
            if epi > batch_size:
                for _ in range(update_iter):
                    loss = learner.update(batch_size)

            if epi % save_interval == 0:
                learner.save_model(model_path)

        print(f"Episode: {epi}, Episode Reward: {np.sum(episode_reward)}, Loss: {loss}")

            

    # env.reset()
    # for agent in range(10000):
    #     actions = [0,1]
    #     # actions = {agent_name: action for agent_name, action in zip(env.agents, actions)}

    #     observation, reward, done, info = env.step(actions)
    #     print(observation[0].shape, reward)
    #     env.render()

    #     if np.any(done):
    #         break