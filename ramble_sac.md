# Ramble on Soft Actor-Critic (SAC)

Recently, the implementation of algorithm Soft Actor-Critic (SAC) makes me aware of some critical problems that I haven't noticed before. 



## Background

The Soft Actor-Critic algorithm is currently state-of-the-art model-free off-policy algorithm for reinforcement learning. 

There are usually three main parts in SAC: the actor (policy) network, the state value network and its target network, and two state-action value networks (with Clipped Double Q-learning trick). The update of SAC can be briefly summarized as: (with implementations in Pytorch )

* For the state-action value networks:

![L(\phi_i, {\mathcal D}) = \underset{(s,a,r,s',d) \sim {\mathcal D}}{{\mathrm E}}\left[     \Bigg( Q_{\phi_i}(s,a) - \left(r + \gamma (1 - d) V_{\psi_{\text{targ}}}(s') \right) \Bigg)^2     \right].](https://spinningup.openai.com/en/latest/_images/math/0a1fc500475d85a71984c03f94462b075da698c3.svg)

​	And its implementation:

```python
predicted_q_value1 = soft_q_net1(state, action)
predicted_q_value2 = soft_q_net2(state, action)
target_value = target_value_net(next_state)
target_q_value = reward + (1 - done) * gamma * target_value
soft_q_criterion1 = nn.MSELoss()
soft_q_criterion2 = nn.MSELoss()
q_value_loss1 = soft_q_criterion1(predicted_q_value1, target_q_value.detach())
q_value_loss2 = soft_q_criterion2(predicted_q_value2, target_q_value.detach())
```



* For the state value network: (with Clipped Double Q-learning trick)

![L(\psi, {\mathcal D}) = \underE{s \sim \mathcal{D} \\ \tilde{a} \sim \pi_{\theta}}{\Bigg(V_{\psi}(s) - \left(\min_{i=1,2} Q_{\phi_i}(s,\tilde{a}) - \alpha \log \pi_{\theta}(\tilde{a}|s) \right)\Bigg)^2}.](https://spinningup.openai.com/en/latest/_images/math/d44a3cfb903ec8530a5880aa718aa5cf0dad28f8.svg)

  	And its implementation:

```python
predicted_new_q_value = torch.min(soft_q_net1(state, new_action),soft_q_net2(state, new_action))
target_value_func = predicted_new_q_value - alpha * log_prob
value_criterion  = nn.MSELoss()
value_loss = value_criterion(predicted_value, target_value_func.detach())
```



* For the policy network: (maximize)

  ![\underE{a \sim \pi}{Q^{\pi}(s,a) - \alpha \log \pi(a|s)}.](https://spinningup.openai.com/en/latest/_images/math/d3deca00a02e211a85278358f902e3ab0683c8a5.svg)

  And its implementation: (three different types)

  ```python
  # 1. Berkely rlkit implementation
  policy_loss = (alpha * log_prob - predicted_new_q_value).mean() 
  # 2. Openai Spinning Up implementation
  policy_loss = (alpha * log_prob - soft_q_net1(state, new_action)).mean()  
  # 3. max Advantage instead of Q to prevent the Q-value drifted high
  policy_loss = (alpha * log_prob - (predicted_new_q_value - predicted_value.detach())).mean() 
  
  ```

  

  

## Quick Facts

* For value-based methods like applying Q(s,a) or V(s), the absolute value doesn't matter, what matters is their relative value.
* Reward normalization can be critical sometimes.
* Sometimes the decreasing policy loss may not indicate better behaviours, several factors may cause that, depending on the format of policy loss functions.



## Details

When I first implemented the soft actor-critic algorithm with the first type of implementation of policy loss, I got a learning curve on my environment like following: (the blue line is reward, and orange line is a  predicted Q value)

![sac (copy)](/home/quantumiracle/research/RL_Robotics2/Soft_Actor_Critic/SAC/sac (copy).png)

<img src="https://github.com/quantumiracle/Reinforcement-Learning-for-Robotics/blob/master/img/3000step_sparse2.pdf" width="80%">

The _alpha_ is set to be 0 to remove the effects of entropy. The reward curve shows that the agent doesn't learn anything useful while the policy loss is actually keep decreasing. This can be derived as the policy loss is negative Q(s,a\_) in this case and the Q(s,a\_) is keep increasing. And it tells us a potential problem in SAC, **if the evaluation of Q and V keeps increasing together, then it will have chances to achieve a small Q loss and small V loss, even more, a decreasing policy loss**. And this is exactly the case happened in my first trial. 

The core of the problem lies in that 

And it is like jump on a soft ground.



Other algorithms like DQN may not suffer from this problem.

* For the Q-learning:

  ![L(\phi, {\mathcal D}) = \underset{(s,a,r,s',d) \sim {\mathcal D}}{{\mathrm E}}\left[     \Bigg( Q_{\phi}(s,a) - \left(r + \gamma (1 - d) \max_{a'} Q_{\phi}(s',a') \right) \Bigg)^2     \right]](https://spinningup.openai.com/en/latest/_images/math/d193a1fae2f39357adc458987f0301518f3cd669.svg)

* For the policy:

  ![\max_{\theta} \underset{s \sim {\mathcal D}}{{\mathrm E}}\left[ Q_{\phi}(s, \mu_{\theta}(s)) \right].](https://spinningup.openai.com/en/latest/_images/math/ec6f2bd53f08da666607434c871e250f86010241.svg)

This two functions cannot be satisfied with only an increasing value estimation of the Q-value network, because the policy loss requires update the policy parameters to maximize the Q-value with fixed Q-value network parameters. (This can be realized with variable\_scope and var\_list in optimizer with Tensorflow or Pytorch) Therefore it provides a solid update for the policy.

A key problem in the Q-value update in DQN is that if the rewards are all positive, the update function will keep drifting the Q-value to be larger and larger. This may not hurt the update of the policy, but will make the update of the Q-value function to chase a moving goal. To solve this problem, the rewards need to be normalized to have mean 0, like with a batch normalization.









