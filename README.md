# State-of-the-art Model-free Reinforcement Learning Algorithms  [![Tweet](https://img.shields.io/twitter/url/http/shields.io.svg?style=social)](https://twitter.com/intent/tweet?text=Get%20over%20170%20free%20design%20blocks%20based%20on%20Bootstrap%204&url=https://www.froala.com/design-blocks&via=froala&hashtags=bootstrap,design,templates,blocks,developers)


PyTorch implemented state-of-the-art model-free reinforcement learning algorithms on both Openai gym environments and a self-implemented Reacher environment. 

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

