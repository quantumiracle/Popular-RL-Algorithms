'''
Probabilistic Mixture-of-Experts
paper: https://arxiv.org/abs/2104.09122

Core features:
It replaces the diagonal Gaussian distribution with (differentiable) Gaussian mixture model for policy function approximation, which is more expressive.
This version is based on on-policy PPO algorithm.
'''

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
# from torch.distributions import Normal
from torch.distributions.normal import Normal
import numpy as np
import wandb
from collections import deque
from torch.utils.tensorboard import SummaryWriter
import argparse
import random

#Hyperparameters
learning_rate = 3e-4
gamma         = 0.99
lmbda         = 0.95
eps_clip      = 0.2
batch_size    = 2048
mini_batch    = int(batch_size//32)
K_epoch       = 10
T_horizon     = 10000
n_epis        = 10000
vf_coef       = 0.5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, default='test',
        help="the name of this experiment")
    parser.add_argument('--wandb_activate', type=bool, default=False, help='whether wandb')
    parser.add_argument("--wandb_entity", type=str, default=None,
        help="the entity (team) of wandb's project")   
    args = parser.parse_args()
    print(args)
    return args

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Actor(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, mix_num):
        super().__init__()
        self.mix_num = mix_num
        self.mean = nn.Sequential(
            layer_init(nn.Linear(num_inputs, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, num_actions * mix_num), std=0.01),
        )
        self.logstd = nn.Parameter(torch.zeros(1, num_actions * mix_num))
        self.mix_coef_linear = nn.Sequential(nn.Linear(num_inputs, mix_num), nn.Softmax(-1))

    def forward(self, x):
        action_mean = self.mean(x)
        action_logstd = self.logstd.expand_as(action_mean)
        mix_coef = self.mix_coef_linear(x)

        return action_mean.reshape(action_mean.shape[0], self.mix_num, -1).squeeze(), \
            action_logstd.reshape(action_logstd.shape[0], self.mix_num, -1).squeeze(), \
            mix_coef.squeeze()

class Critic(nn.Module):
    def __init__(self, num_inputs, hidden_dim):
        super().__init__()
        self.model = nn.Sequential(
            layer_init(nn.Linear(num_inputs, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )

    def forward(self, x):
        return self.model(x)

class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, init_w=3e-3):
        super(QNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim)
        self.linear4 = nn.Linear(hidden_dim, 1)

        self.linear4.weight.data.uniform_(-init_w, init_w)
        self.linear4.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action, match_shape=False):
        if match_shape:
            state = state.unsqueeze(1).repeat(1, action.shape[1], 1)
        x = torch.cat([state, action], -1)  # the dim 0 is number of samples
        x = x.reshape(-1, x.shape[-1])
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.linear4(x)
        if match_shape:
            return x.reshape(-1, action.shape[1])
        else:
            return x

class PMOE_PPO():
    def __init__(self, num_inputs, num_actions, hidden_dim, mix_num=5):
        self.data = deque(maxlen=batch_size)  # a ring buffer
        self.max_grad_norm = 0.5
        self.v_loss_clip = True
        self.mix_num = mix_num

        self.critic = Critic(num_inputs, hidden_dim).to(device)
        self.actor = Actor(num_inputs, num_actions, hidden_dim, mix_num).to(device)
        self.q_net = QNetwork(num_inputs, num_actions, hidden_dim).to(device)

        self.parameters = list(self.critic.parameters()) + list(self.actor.parameters()) + list(self.q_net.parameters())
        self.optimizer = optim.Adam(self.parameters, lr=learning_rate, eps=1e-5)

    def pi(self, x):
        return self.actor(x)
    
    def v(self, x):
        return self.critic(x)
      
    def get_action_and_value(self, x, action=None, select_from_mixture=True, track_grad=False):
        mean, log_std, mix_coef = self.pi(x)
        std = torch.exp(log_std)
        normal = Normal(mean, std)
        
        full_action = normal.sample()
        if select_from_mixture:
            mix_dist = Categorical(mix_coef)
            index = mix_dist.sample()
            a = full_action[index]
        else:
            a = full_action

        if action is None:
            a_for_prob = a.unsqueeze(-2)  # to (1, action_dim), matching with mean and std (K, action_dim)
            log_prob = (mix_coef @ normal.log_prob(a_for_prob).sum(-1).exp()).log()
        else: # work for batch
            a_for_prob = action.unsqueeze(-2) # use given action for calculating probability
            log_prob = torch.einsum('ij,ij->i', mix_coef, normal.log_prob(a_for_prob).sum(-1).exp()).log()   # prob of action from the whole GMM, including the mixing
        a_for_prob = a_for_prob
            
        value =  self.v(x)
        if track_grad:
            return a, log_prob, value, mix_coef
        else:
            return a.cpu().numpy(), log_prob, value, mix_coef

    def put_data(self, transition):
        self.data.append(transition)

    def make_batch(self,):
        s, a, r, s_prime, logprob_a, v, done_mask = zip(*self.data)
        s,a,r,s_prime,logprob_a,v,done_mask = torch.tensor(np.array(s), dtype=torch.float), torch.tensor(np.array(a)), \
                                          torch.tensor(np.array(r), dtype=torch.float).unsqueeze(-1), torch.tensor(np.array(s_prime), dtype=torch.float), \
                                        torch.tensor(logprob_a).unsqueeze(-1), torch.tensor(v).unsqueeze(-1), torch.tensor(np.array(done_mask), dtype=torch.float).unsqueeze(-1)
        return s.to(device), a.to(device), r.to(device), s_prime.to(device), done_mask.to(device), logprob_a.to(device), v.to(device)
        
    def train_net(self):
        s, a, r, s_prime, done_mask, logprob_a, v = self.make_batch()
        loss_list = []
        with torch.no_grad():
            advantage = torch.zeros_like(r).to(device)
            lastgaelam = 0
            for t in reversed(range(s.shape[0])):
                if done_mask[t] or t == s.shape[0]-1:
                    nextvalues = self.v(s_prime[t])
                else:
                    nextvalues = v[t+1]
                delta = r[t] + gamma * nextvalues * (1.0-done_mask[t]) - v[t]
                advantage[t] = lastgaelam = delta + gamma * lmbda * lastgaelam * (1.0-done_mask[t])
            assert advantage.shape == v.shape
            td_target = advantage + v

        # minibatch SGD over the entire buffer (K epochs)
        b_inds = np.arange(batch_size)
        for epoch in range(K_epoch):    
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, mini_batch):
                end = start + mini_batch
                minibatch_idx = b_inds[start:end]

                bs, ba, blogprob_a, bv = s[minibatch_idx], a[minibatch_idx], logprob_a[minibatch_idx].reshape(-1), v[minibatch_idx].reshape(-1)
                badvantage, btd_target = advantage[minibatch_idx].reshape(-1), td_target[minibatch_idx].reshape(-1)

                if not torch.isnan(badvantage.std()):
                    badvantage = (badvantage - badvantage.mean()) / (badvantage.std() + 1e-8) 
        
                # get mixing coefficients loss
                new_a, newlogprob_a, new_vs, new_mix_coef = self.get_action_and_value(bs, ba, select_from_mixture=False, track_grad=True)
                new_q = self.q_net(bs, new_a, match_shape=True)
                _, best_index = new_q.max(-1)
                coef_loss = F.mse_loss(new_mix_coef, F.one_hot(best_index, self.mix_num).float()).mean()

                # Q-net loss
                pred_q = self.q_net(bs, ba).squeeze()
                q_loss = F.mse_loss(pred_q, btd_target).mean()

                new_vs = new_vs.reshape(-1)
                ratio = torch.exp(newlogprob_a - blogprob_a)  # a/b == exp(log(a)-log(b))
                surr1 = -ratio * badvantage
                surr2 = -torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * badvantage
                policy_loss = torch.max(surr1, surr2).mean()
                # import pdb; pdb.set_trace()

                if self.v_loss_clip: # clipped value loss
                    v_clipped = bv + torch.clamp(new_vs - bv, -eps_clip, eps_clip)
                    value_loss_clipped = (v_clipped - btd_target) ** 2
                    value_loss_unclipped = (new_vs - btd_target) ** 2
                    value_loss_max = torch.max(value_loss_unclipped, value_loss_clipped)
                    value_loss =  0.5 * value_loss_max.mean()
                else:
                    value_loss = F.smooth_l1_loss(new_vs, btd_target)

                loss = coef_loss + policy_loss + vf_coef * value_loss + q_loss
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters, self.max_grad_norm)
                self.optimizer.step()
        loss_list = [coef_loss.item(), q_loss.item(), policy_loss.item(), value_loss.item()]
        return loss_list
        
def main():
    args = parse_args()
    env_id = 2
    seed = 1
    env_name = ['HalfCheetah-v2', 'Ant-v2', 'Hopper-v2'][env_id]
    env = gym.make(env_name)
    env = gym.wrappers.RecordEpisodeStatistics(env) # bypass the reward normalization to record episodic return
    env = gym.wrappers.ClipAction(env)
    env = gym.wrappers.NormalizeObservation(env)
    env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
    env = gym.wrappers.NormalizeReward(env) # this improves learning significantly
    env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    print(env.observation_space, env.action_space)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    state_dim = env.observation_space.shape[0]
    action_dim =  env.action_space.shape[0]
    hidden_dim = 64
    mix_num = 5 # number of experts
    model = PMOE_PPO(state_dim, action_dim, hidden_dim, mix_num)
    score = 0.0
    print_interval = 1
    step = 0
    update = 1
    loss_list = []

    if args.wandb_activate:
        wandb.init(
            project=args.run,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=args.run+f'_{env_name}',
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/pmoe-ppo_{args.run}_{env_name}")

    for n_epi in range(n_epis):
        s = env.reset()
        done = False
        epi_r = 0.
        ## learning rate schedule
        # frac = 1.0 - (n_epi - 1.0) / n_epis
        # lrnow = frac * learning_rate
        # model.optimizer.param_groups[0]["lr"] = lrnow

        # while not done:
        for t in range(T_horizon):
            step += 1
            with torch.no_grad():
                a, logprob, v, _ = model.get_action_and_value(torch.from_numpy(s).float().unsqueeze(0).to(device))
            s_prime, r, done, info = env.step(a)
            # env.render()

            model.put_data((s, a, r, s_prime, logprob, v.squeeze(-1), done))

            s = s_prime

            score += r

            if step % batch_size == 0 and step > 0:
                loss_list = model.train_net()
                update += 1
                eff_update = update
                
            if 'episode' in info.keys():
                epi_r = info['episode']['r']
                print(f"Global steps: {step}, score: {epi_r}")
            
            if done:
                break

        if n_epi%print_interval==0 and n_epi!=0:
            # print("Global steps: {}, # of episode :{}, avg score : {:.1f}".format(step, n_epi, score/print_interval)) # this is normalized reward
            writer.add_scalar("charts/episodic_return", epi_r, n_epi)
            writer.add_scalar("charts/episodic_length", t, n_epi)
            writer.add_scalar("charts/update", update, n_epi)
            writer.add_scalar("charts/", update, n_epi)
            if len(loss_list) > 0:
                writer.add_scalar("charts/coeff_loss", loss_list[0], n_epi)
                writer.add_scalar("charts/Q_loss", loss_list[1], n_epi)
                writer.add_scalar("charts/policy_loss", loss_list[2], n_epi)
                writer.add_scalar("charts/value_loss", loss_list[3], n_epi)

            score = 0.0

    env.close()

if __name__ == '__main__':
    main()