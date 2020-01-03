# State-of-the-art Model-free Reinforcement Learning Algorithms  [![Tweet](https://img.shields.io/twitter/url/http/shields.io.svg?style=social)](https://twitter.com/intent/tweet?text=State-of-the-art-Model-free-Reinforcement-Learning-Algorithms%20&url=hhttps://github.com/quantumiracle/STOA-RL-Algorithms&hashtags=RL)


**PyTorch** and **Tensorflow 2.0** implementation of state-of-the-art model-free reinforcement learning algorithms on both Openai gym environments and a self-implemented Reacher environment. 

Algorithms include **Soft Actor-Critic (SAC), Deep Deterministic Policy Gradient (DDPG), Twin Delayed DDPG (TD3), Actor-Critic (AC/A2C), Proximal Policy Optimization (PPO), QT-Opt (including Cross-entropy (CE) Method)**, **PointNet**, **Transporter**, **Recurrent Policy Gradient**, etc.

This repo only contains **PyTorch** Implementation.

[**Here**](https://github.com/tensorlayer/tensorlayer/tree/reinforcement-learning/examples/reinforcement_learning) is my **Tensorflow 2.0 + Tensorlayer 2.0** implementation as tutorials with simple structures. And [**here**](https://github.com/tensorlayer/RLzoo) is a baseline implementation with high-level API supporting a variety of popular environments, also with Tensorflow 2.0 + Tensorlayer 2.0.

## Contents:

* Two versions of **Soft Actor-Critic (SAC)** are implemented.

  **SAC Version 1**:

     `sac.py`: using state-value function.

     paper: https://arxiv.org/pdf/1801.01290.pdf

  **SAC Version 2**:

   `sac_v2.py`: using target Q-value function instead of state-value function.

    paper: https://arxiv.org/pdf/1812.05905.pdf

* **Deep Deterministic Policy Gradient (DDPG)**:

  `ddpg.py`: implementation of DDPG.

* **Twin Delayed DDPG (TD3)**:

   `td3.py`: implementation of TD3.

   paper: https://arxiv.org/pdf/1802.09477.pdf

* **Proximal Policy Optimization (PPO)**:
  Todo

* **Actor-Critic (AC) / A2C**:

  `ac.py`: extensible AC/A2C, easy to change to be DDPG, etc.

   A very extensible version of vanilla AC/A2C, supporting for all continuous/discrete deterministic/non-deterministic cases.

* Two versions of **QT-Opt** are implemented [here](https://github.com/quantumiracle/QT_Opt).

* **PointNet** for landmarks generation from images with unsupervised learning is implemented [here](https://github.com/quantumiracle/PointNet_Landmarks_from_Image/tree/master). This method is also used for image-based reinforcement learning as a SOTA algorithm, called **Transporter**.

  original paper: [Unsupervised Learning of Object Landmarksthrough Conditional Image Generation](https://papers.nips.cc/paper/7657-unsupervised-learning-of-object-landmarks-through-conditional-image-generation.pdf)

  paper for RL: [Unsupervised Learning of Object Keypointsfor Perception and Control](https://arxiv.org/pdf/1906.11883.pdf)

* **Recurrent Policy Gradient**:

  `rdpg.py`: DDPG with LSTM policy.

  `td3_lstm.py`: TD3 with LSTM policy.

  `sac_v2_lstm.py`: SAC with LSTM policy.

  References:

  [Memory-based control with recurrent neural networks](https://arxiv.org/abs/1512.04455)

  [Sim-to-Real Transfer of Robotic Control with Dynamics Randomization](https://arxiv.org/abs/1710.06537)
  
 * **Maximum a Posteriori Policy Optimisation (MPO)**:
 
    todo

    paper: [Maximum a Posteriori Policy Optimisation](https://arxiv.org/abs/1806.06920)
 
 * **Advantage-Weighted Regression (AWR)**:

    todo 

    paper: [Advantage-Weighted Regression: Simple and Scalable Off-Policy Reinforcement Learning](https://arxiv.org/pdf/1910.00177.pdf)

## Usage:
`python ***.py --train` 

`python ***.py --test` 

## Troubleshooting:

If you meet problem *"Not imlplemented Error"*, it may be due to the wrong gym version. The newest gym==0.14 won't work. Install gym==0.7 or gym==0.10 with `pip install -r requirements.txt`.

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

## Citation:
To cite this repository:
```
@misc{rlalgorithms,
  author = {Zihan Ding},
  title = {SOTA-RL-Algorithms},
  year = {2019},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/quantumiracle/SOTA-RL-Algorithms}},
}
```
