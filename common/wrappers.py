
import numpy as np
import gym


class Dict2TupleWrapper():
    """ Wrap the PettingZoo envs to have a similar style as LaserFrame in NFSP """
    def __init__(self, env, keep_info=False):
        super(Dict2TupleWrapper, self).__init__()
        self.env = env
        self.num_agents = env.num_agents
        self.keep_info = keep_info  # if True keep info as dict
        if len(env.observation_space.shape) > 1: # image
            old_shape = env.observation_space.shape
            self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(old_shape[-1], old_shape[0], old_shape[1]), dtype=np.uint8)
            self.obs_type = 'rgb_image'
        else:
            self.observation_space = env.observation_space
            self.obs_type = 'ram'
        self.action_space = env.action_space
        self.observation_spaces = env.observation_spaces
        self.action_spaces = env.action_spaces
        try:   # both pettingzoo and slimevolley can work with this
            self.agents = env.agents
        except:
            self.agents = env.unwrapped.agents
    
    @property
    def unwrapped(self,):
        return self.env

    @property
    def spec(self):
        return self.env.spec

    def observation_swapaxis(self, observation):
        return (np.swapaxes(observation[0], 2, 0), np.swapaxes(observation[1], 2, 0))
    
    def reset(self):
        obs_dict = self.env.reset()
        if self.obs_type == 'ram':
            return tuple(obs_dict.values())
        else:
            return self.observation_swapaxis(tuple(obs_dict.values()))

    def step(self, actions): 
        actions = {agent_name: action for agent_name, action in zip(self.agents, actions)}
        obs, rewards, dones, infos = self.env.step(actions)
        if self.obs_type == 'ram':
            o = tuple(obs.values())
        else:
            o = self.observation_swapaxis(tuple(obs.values()))
        r = list(rewards.values())
        d = list(dones.values())
        if self.keep_info:  # a special case for VectorEnv
            info = infos
        else:
            info = list(infos.values())
        del obs,rewards, dones, infos
        # r = self._zerosum_filter(r)

        return o, r, d, info

    def _zerosum_filter(self, r):
        ## zero-sum filter: 
        # added for making non-zero sum game to be zero-sum, e.g. tennis_v2
        if np.sum(r) != 0:
            nonzero_idx = np.nonzero(r)[0][0]
            r[1-nonzero_idx] = -r[nonzero_idx]
        return r

    def seed(self, seed):
        self.env.seed(seed)
        np.random.seed(seed)

    def render(self,):
        self.env.render()

    def close(self):
        self.env.close()