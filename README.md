# Popular Model-free Reinforcement Learning Algorithms  [![Tweet](https://img.shields.io/twitter/url/http/shields.io.svg?style=social)](https://twitter.com/intent/tweet?text=State-of-the-art-Model-free-Reinforcement-Learning-Algorithms%20&url=hhttps://github.com/quantumiracle/STOA-RL-Algorithms&hashtags=RL)


**PyTorch** and **Tensorflow 2.0** implementation of state-of-the-art model-free reinforcement learning algorithms on both Openai gym environments and a self-implemented Reacher environment. 

Algorithms include **Soft Actor-Critic (SAC), Deep Deterministic Policy Gradient (DDPG), Twin Delayed DDPG (TD3), Actor-Critic (AC/A2C), Proximal Policy Optimization (PPO), QT-Opt (including Cross-entropy (CE) Method)**, **PointNet**, **Transporter**, **Recurrent Policy Gradient**, **Soft Decision Tree**, **Probabilistic Mixture-of-Experts**, etc.

Please note that this repo is more of a personal collection of algorithms I implemented and tested during my research and study period, rather than an official open-source library/package for usage. However, I think it could be helpful to share it with others and I'm expecting useful discussions on my implementations. But I didn't spend much time on cleaning or structuring the code. As you may notice that there may be several versions of implementation for each algorithm, I intentionally show all of them here for you to refer and compare. Also, this repo contains only **PyTorch** Implementation.

For official libraries of RL algorithms, I provided the following two with **TensorFlow 2.0 + TensorLayer 2.0**:

* [**RL Tutorial**](https://github.com/tensorlayer/tensorlayer/tree/reinforcement-learning/examples/reinforcement_learning) (*Status: Released*) contains RL algorithms implementation as tutorials with simple structures. 

* [**RLzoo**](https://github.com/tensorlayer/RLzoo) (*Status: Released*) is a baseline implementation with high-level API supporting a variety of popular environments, with more hierarchical structures for simple usage.

Since Tensorflow 2.0 has already incorporated the dynamic graph construction instead of the static one, it becomes a trivial work to transfer the RL code between TensorFlow and PyTorch.

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
  
  For continuous environments, two versions are implemented:
  
  Version 1: `ppo_continuous.py` and `ppo_continuous_multiprocess.py` 
  
  Version 2: `ppo_continuous2.py` and `ppo_continuous_multiprocess2.py` 
  
  For discrete environment:
  
  `ppo_gae_discrete.py`: with Generalized Advantage Estimation (GAE)

* **Actor-Critic (AC) / A2C**:

  `ac.py`: extensible AC/A2C, easy to change to be DDPG, etc.

   A very extensible version of vanilla AC/A2C, supporting for all continuous/discrete deterministic/non-deterministic cases.

* **QT-Opt**:

   Two versions are implemented [here](https://github.com/quantumiracle/QT_Opt).

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
  
 * **Soft Decision Tree** as function approximator for PPO:
 
   `sdt_ppo_gae_discrete.py`: replace the network layers of policy in PPO to be a [Soft Decision Tree](https://arxiv.org/abs/1711.09784), for achieving explainable RL.
   
   paper: [CDT: Cascading Decision Trees for Explainable Reinforcement Learning
](https://arxiv.org/abs/2011.07553)
   
 * **Probabilistic Mixture-of-Experts (PMOE)** :
 
   `pmoe.py`: uses a differentiable multi-modal Gaussian distribution to replace the standard unimodal Gaussian distribution for policy representation.
   
   paper: [Probabilistic Mixture-of-Experts for Efficient Deep Reinforcement Learning
](https://arxiv.org/pdf/2104.09122)

 
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

## Undervalued tricks:

As we all known, there are various tricks in empirical RL algorithm implementations in support the performance in practice, including hyper-parameters, normalization, network architecture or even hidden activation function, etc. I summarize some I met with the programs in this repo here:

 * Environment specific: 
    *  For `Pendulum-v0` environment in Gym, a reward pre-processing as `(r+8)/8` usually improves the learning efficiency, as [here](https://github.com/quantumiracle/Popular-RL-Algorithms/blob/7f2bb74a51cf9cbde92a6ccfa42e97dc129dd145/ppo_continuous2.py#L376)
    Also, this environment needs the [maximum episode length](https://github.com/quantumiracle/Popular-RL-Algorithms/blob/7f2bb74a51cf9cbde92a6ccfa42e97dc129dd145/sac_v2.py#L364) to be at least 150 to learn well, too short episodes make it hard to learn.
    * `MountainCar-v0` environment in Gym has very sparse reward (only when reaching the flag), general learning curves will be noisy; therefore some specific process may also need for this environment.

 * Normalization:
    * [Reward normalization](https://github.com/quantumiracle/Popular-RL-Algorithms/blob/7f2bb74a51cf9cbde92a6ccfa42e97dc129dd145/sac_v2.py#L262) or [advantage normalization](https://github.com/quantumiracle/Popular-RL-Algorithms/blob/881903e4aa22921f142daedfcf3dd266488405d8/ppo_gae_discrete.py#L79) in batch can have great improvements on performance (learning efficiency, stability) sometimes, although theoretically on-policy algorithms like PPO should not apply data normalization during training due to distribution shift. For an in-depth look at this problem, we should treat it differently (1) when normalizing the direct input data like observation, action, reward, etc; (2) when normalizing the estimation of the values (state value, state-action value, advantage, etc). For (1), a more reasonable way for normalization is to keep a moving average of previous mean and standard deviation, to achieve a similar effect as conducting the normaliztation on the full dataset during RL agent learning (this is not possible since in RL the data comes from interaction of agents and environments). For (2), we can simply conduct normalization on value estimations (rather than keeping the historical average) since we do not want the estimated values to have distribution shift, so we treat them like a static distribution.

* Although I provide the multiprocessing versions of serveral algorithms ([SAC](https://github.com/quantumiracle/Popular-RL-Algorithms/blob/master/sac_v2_multiprocess.py), [PPO](https://github.com/quantumiracle/Popular-RL-Algorithms/blob/master/ppo_continuous_multiprocess2.py), etc), for small-scale environments in Gym, this is usually not necessary or even inefficient. The vectorized environment wrapper for parallel environment sampling may be more proper solution for learning these environments, since the bottelneck in learning efficiency mainly lies in the interaction with environments rather than the model learning (back-propagation) process.

More discussions about **implementation tricks** see this [chapter](https://link.springer.com/chapter/10.1007/978-981-15-4095-0_18) in our book.

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
  title = {Popular-RL-Algorithms},
  year = {2019},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/quantumiracle/Popular-RL-Algorithms}},
}
```

## Other Resources:
<p align="center">
<img src="https://github.com/quantumiracle/STOA-RL-Algorithms/blob/master/img/drl_book.png" width="20%">
</p>

**Deep Reinforcement Learning: Foundamentals, Research and Applications** *Springer Nature 2020* 

is the book I edited with [Dr. Hao Dong](https://github.com/zsdonghao) and [Dr. Shanghang Zhang](https://scholar.google.com/citations?user=voqw10cAAAAJ&hl=en), which provides a wide coverage of topics in deep reinforcement learning. Details see [website](https://deepreinforcementlearningbook.org/) and [Springer webpage](https://www.springer.com/gp/book/9789811540943). To cite the book:
```
@book{deepRL-2020,
 title={Deep Reinforcement Learning: Fundamentals, Research, and Applications},
 editor={Hao Dong, Zihan Ding, Shanghang Zhang},
 author={Hao Dong, Zihan Ding, Shanghang Zhang, Hang Yuan, Hongming Zhang, Jingqing Zhang, Yanhua Huang, Tianyang Yu, Huaqing Zhang, Ruitong Huang},
 publisher={Springer Nature},
 note={\url{http://www.deepreinforcementlearningbook.org}},
 year={2020}
}
```
