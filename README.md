# State-of-the-art Model-free Reinforcement Learning Algorithms  [![Tweet](https://img.shields.io/twitter/url/http/shields.io.svg?style=social)](https://twitter.com/intent/tweet?text=State-of-the-art-Model-free-Reinforcement-Learning-Algorithms%20&url=hhttps://github.com/quantumiracle/STOA-RL-Algorithms&hashtags=RL)


PyTorch implemented state-of-the-art model-free reinforcement learning algorithms on both Openai gym environments and a self-implemented Reacher environment. 

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
* **AC** for gym CartPole-v0:
<p align="center">
<img src="https://github.com/quantumiracle/STOA-RL-Algorithms/blob/master/ac_cartpole.png" width="100%">
</p>
        However, vanilla AC/A2C cannot handle the continuous case like gym Pendulum-v0 well.


* Both two versions of **SAC** seem to have better performance, compared with **TD3**.
