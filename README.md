# State-of-the-art Model-free Reinforcement Learning Algorithms 

PyTorch implemented state-of-the-art model-free reinforcement learning algorithms on both Openai gym environments and a self-implemented Reacher environment (any number of joints or link lengths or target positions). 

Two versions of SAC are implemented.

Version 1:

* `sac.py`: using state-value function.

  paper: https://arxiv.org/pdf/1801.01290.pdf

Version 2:

* `sac_v2.py`: using target Q-value function instead of state-value function.

  paper: https://arxiv.org/pdf/1812.05905.pdf

TD3: Twin Delayed DDPG.

* `td3.py`:

  paper: https://arxiv.org/pdf/1802.09477.pdf

PPO:

Actor-Critic:

