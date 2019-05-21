# State-of-the-art Model-free Reinforcement Learning Algorithms  [![Tweet](https://img.shields.io/twitter/url/http/shields.io.svg?style=social)](https://twitter.com/intent/tweet?text=State-of-the-art-Model-free-Reinforcement-Learning-Algorithms%20&url=hhttps://github.com/quantumiracle/STOA-RL-Algorithms&hashtags=RL)


**PyTorch** implementation of state-of-the-art model-free reinforcement learning algorithms on both Openai gym environments and a self-implemented Reacher environment. Algorithms include Soft Actor-Critic (SAC), Twin Delayed DDPG (TD3), Actor-Critic (AC/A2C), Proximal Policy Optimization (PPO), etc.

[Here](https://github.com/tensorlayer/tensorlayer/tree/reinforcement-learning/examples/reinforcement_learning) is my **Tensorflow 2.0 + Tensorlayer 2.0** implementation. 

## Contents:

Two versions of **SAC** are implemented.

**SAC Version 1**:

* `sac.py`: using state-value function.

  paper: https://arxiv.org/pdf/1801.01290.pdf

**SAC Version 2**:

* `sac_v2.py`: using target Q-value function instead of state-value function.

  paper: https://arxiv.org/pdf/1812.05905.pdf

**TD3: Twin Delayed DDPG**:

* `td3.py`:

  paper: https://arxiv.org/pdf/1802.09477.pdf

**PPO**:

**Actor-Critic (AC) / A2C**:

* `ac.py`: extensible AC/A2C, easy to change to be DDPG, etc.

  A very extensible version of vanilla AC/A2C, supporting for all continuous/discrete deterministic/non-deterministic cases.


## Usage:
`python ***.py`

## Performance:
* **SAC** for gym Pendulum-v0:

SAC with automatically updating variable alpha for entropy:
<p align="center">
<img src="https://github.com/quantumiracle/STOA-RL-Algorithms/blob/master/img/sac_autoentropy.png" width="100%">
</p>
SAC without automatically updating variable alpha for entropy:
<p align="center">
<img src="https://github.com/quantumiracle/STOA-RL-Algorithms/blob/master/img/sac_nonautoentropy.png" width="100%">
</p>

It shows that the automatic-entropy update helps the agent to learn faster.

* **TD3** for gym Pendulum-v0:

TD3 with deterministic policy:
<p align="center">
<img src="https://github.com/quantumiracle/STOA-RL-Algorithms/blob/master/img/td3_deterministic.png" width="100%">
</p>
TD3 with non-deterministic/stochastic policy:
<p align="center">
<img src="https://github.com/quantumiracle/STOA-RL-Algorithms/blob/master/img/td3_nondeterministic.png" width="100%">
</p>

It seems TD3 with deterministic policy works a little better, but basically similar.

* **AC** for gym CartPole-v0:
<p align="center">
<img src="https://github.com/quantumiracle/STOA-RL-Algorithms/blob/master/img/ac_cartpole.png" width="100%">
</p>

   However, vanilla AC/A2C cannot handle the continuous case like gym Pendulum-v0 well.

