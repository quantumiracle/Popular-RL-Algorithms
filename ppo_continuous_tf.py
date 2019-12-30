import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import gym
from reacher import Reacher
import argparse
tf.random.set_random_seed(123)
EP_MAX = 10000
EP_LEN = 20
GAMMA = 0.9
A_LR = 1e-4
C_LR = 2e-4
BATCH = 64
A_UPDATE_STEPS = 10
C_UPDATE_STEPS = 10
S_DIM, A_DIM = 8,2
METHOD = [
    dict(name='kl_pen', kl_target=0.01, lam=0.5),   # KL penalty
    dict(name='clip', epsilon=0.2),                 # Clipped surrogate objective, find this is better
][1]        # choose the method for optimization


model_path='./ppo_model/ppo'

parser = argparse.ArgumentParser(description='Train or test neural net motor controller.')
parser.add_argument('--train', dest='train', action='store_true', default=False)
parser.add_argument('--test', dest='test', action='store_true', default=True)
args = parser.parse_args()

class PPO(object):

    def __init__(self):
        self.sess = tf.Session()
        self.tfs = tf.placeholder(tf.float32, [None, S_DIM], 'state')

        # critic
        with tf.variable_scope('critic'):
            l1 = tf.layers.dense(self.tfs, 100, tf.nn.relu)
            self.v = tf.layers.dense(l1, 1)
            self.tfdc_r = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
            self.advantage = self.tfdc_r - self.v
            self.closs = tf.reduce_mean(tf.square(self.advantage))
            self.ctrain_op = tf.train.AdamOptimizer(C_LR).minimize(self.closs)

        # actor
        pi, pi_params = self._build_anet('pi', trainable=True)
        oldpi, oldpi_params = self._build_anet('oldpi', trainable=False)
        with tf.variable_scope('sample_action'):
            self.sample_op = tf.squeeze(pi.sample(1), axis=0)       # choosing action
        with tf.variable_scope('update_oldpi'):
            self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]

        self.tfa = tf.placeholder(tf.float32, [None, A_DIM], 'action')
        self.tfadv = tf.placeholder(tf.float32, [None, 1], 'advantage')
        with tf.variable_scope('loss'):
            with tf.variable_scope('surrogate'):
                ratio = tf.exp(pi.log_prob(self.tfa) - oldpi.log_prob(self.tfa))
                # ratio = pi.prob(self.tfa) / (oldpi.prob(self.tfa)+1e-6)  # avoid the numerical problem: NAN
                surr = ratio * self.tfadv
            if METHOD['name'] == 'kl_pen':
                self.tflam = tf.placeholder(tf.float32, None, 'lambda')
                kl = tf.distributions.kl_divergence(oldpi, pi)
                self.kl_mean = tf.reduce_mean(kl)
                self.aloss = -(tf.reduce_mean(surr - self.tflam * kl))
            else:   # clipping method, find this is better
                self.aloss = -tf.reduce_mean(tf.minimum(
                    surr,
                    tf.clip_by_value(ratio, 1.-METHOD['epsilon'], 1.+METHOD['epsilon'])*self.tfadv))
            # add entropy to boost exploration
            # entropy=pi.entropy()
            # self.aloss-=0.1*entropy
        with tf.variable_scope('atrain'):
            self.atrain_op = tf.train.AdamOptimizer(A_LR).minimize(self.aloss)

        tf.summary.FileWriter("log/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

    def update(self, s, a, r):
        self.sess.run(self.update_oldpi_op)
        adv = self.sess.run(self.advantage, {self.tfs: s, self.tfdc_r: r})
        adv = (adv - adv.mean())/(adv.std()+1e-6)     # sometimes helpful

        # update actor
        if METHOD['name'] == 'kl_pen':
            for i in range(A_UPDATE_STEPS):
                print('updata: ',i)
                _, kl = self.sess.run(
                    [self.atrain_op, self.kl_mean],
                    {self.tfs: s, self.tfa: a, self.tfadv: adv, self.tflam: METHOD['lam']})
                if kl > 4*METHOD['kl_target']:  # this in in google's paper
                    break
            if kl < METHOD['kl_target'] / 1.5:  # adaptive lambda, this is in OpenAI's paper
                METHOD['lam'] /= 2
            elif kl > METHOD['kl_target'] * 1.5:
                METHOD['lam'] *= 2
            METHOD['lam'] = np.clip(METHOD['lam'], 1e-4, 10)    # sometimes explode, this clipping is my solution
        else:   # clipping method, find this is better (OpenAI's paper)
            [self.sess.run(self.atrain_op, {self.tfs: s, self.tfa: a, self.tfadv: adv}) for _ in range(A_UPDATE_STEPS)]

        # update critic
        [self.sess.run(self.ctrain_op, {self.tfs: s, self.tfdc_r: r}) for _ in range(C_UPDATE_STEPS)]

    def _build_anet(self, name, trainable):
        with tf.variable_scope(name):
            l1 = tf.layers.dense(self.tfs, 100, tf.nn.relu, trainable=trainable)
            # l1 = tf.layers.batch_normalization(tf.layers.dense(self.tfs, 100, tf.nn.relu, trainable=trainable), training=True)
            '''the action mean mu is set to be scale 10 instead of 360, avoiding useless shaking and one-step to goal!'''
            # mu =  10.*tf.layers.dense(l1, A_DIM, tf.nn.tanh, trainable=trainable)
            mu = tf.layers.dense(l1, A_DIM, tf.nn.leaky_relu, trainable=trainable)
            # sigma = tf.layers.dense(l1, A_DIM, tf.nn.softplus, trainable=trainable) # softplus to make it positive
            sigma = tf.layers.dense(l1, A_DIM, tf.nn.sigmoid, trainable=trainable) # softplus to make it positive

            # in case that sigma is 0
            sigma +=5e-1 # 5e-1 for boosting exploration, accelerating learning.
            self.mu=mu
            self.sigma=sigma
            norm_dist = tf.distributions.Normal(loc=mu, scale=sigma)

        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return norm_dist, params

    def choose_action(self, s):
        s = s[np.newaxis, :]
        a ,mu, sigma= self.sess.run([self.sample_op, self.mu, self.sigma], {self.tfs: s})
        # print('s: ',s)
        # print('a: ', a)
        # print('mu, sigma: ', mu,sigma)
        return np.clip(a[0], -360, 360)

    def get_v(self, s):
        if s.ndim < 2: s = s[np.newaxis, :]
        return self.sess.run(self.v, {self.tfs: s})[0, 0]

    def save(self, path):
        saver = tf.train.Saver()
        saver.save(self.sess, path)

    def load(self, path):
        saver=tf.train.Saver()
        saver.restore(self.sess, path)
        
NUM_JOINTS=2
LINK_LENGTH=[200, 140]
INI_JOING_ANGLES=[0.1, 0.1]
SCREEN_SIZE=1000
SPARSE_REWARD=False
SCREEN_SHOT=False
DETERMINISTIC=False
env=Reacher(screen_size=SCREEN_SIZE, num_joints=NUM_JOINTS, link_lengths = LINK_LENGTH, \
    ini_joint_angles=INI_JOING_ANGLES, target_pos = [369,430], render=True)
ppo = PPO()

if args.train:
    all_ep_r = []

    for ep in range(EP_MAX):
        s = env.reset(SCREEN_SHOT)
        buffer_s, buffer_a, buffer_r = [], [], []
        ep_r = 0
        for t in range(EP_LEN):    # in one episode
            # env.render()
            a = ppo.choose_action(s)
            s_, r, done, distance2goal = env.step(a, SPARSE_REWARD, SCREEN_SHOT)
            buffer_s.append(s)
            buffer_a.append(a)
            # print('r, norm_r: ', r, (r+8)/8)
            '''the normalization makes reacher's reward almost same and not work'''
            # buffer_r.append((r+8)/8)    # normalize reward, find to be useful
            buffer_r.append(r)
            s = s_
            ep_r += r

            # update ppo
            if (t+1) % BATCH == 0 or t == EP_LEN-1:
                v_s_ = ppo.get_v(s_)
                discounted_r = []
                for r in buffer_r[::-1]:
                    v_s_ = r + GAMMA * v_s_
                    discounted_r.append(v_s_)
                discounted_r.reverse()

                bs, ba, br = np.vstack(buffer_s), np.vstack(buffer_a), np.array(discounted_r)[:, np.newaxis]
                buffer_s, buffer_a, buffer_r = [], [], []
                ppo.update(bs, ba, br)
        if ep == 0: all_ep_r.append(ep_r)
        else: all_ep_r.append(all_ep_r[-1]*0.9 + ep_r*0.1)
        print(
            'Ep: %i' % ep,
            "|Ep_r: %.4f" % ep_r,
            '|Dis2Goal: %.4f' %distance2goal,
            ("|Lam: %.4f" % METHOD['lam']) if METHOD['name'] == 'kl_pen' else '',
        )
        if ep % 500==0:
            ppo.save(model_path)
            plt.plot(np.arange(len(all_ep_r)), all_ep_r)
            plt.xlabel('Episode');plt.ylabel('Moving averaged episode reward');plt.savefig('./ppo_single_2.png')

if args.test:
    all_ep_r = []
    ppo.load(model_path)
    EP_MAX = 1000
    print('-------------- TEST --------------- ')
    for ep in range(EP_MAX):
        print('Episode: ', ep)
        s = env.reset()
        buffer_s, buffer_a, buffer_r = [], [], []
        ep_r = 0
        for t in range(EP_LEN):    # in one episode
            a = ppo.choose_action(s)
            s_, r, done = env.step(a)
            buffer_s.append(s)
            buffer_a.append(a)
            # print('r, norm_r: ', r, (r+8)/8)
            '''the normalization makes reacher's reward almost same and not work'''
            # buffer_r.append((r+8)/8)    # normalize reward, find to be useful
            buffer_r.append(r)
            s = s_
            ep_r += r
            # no update

        all_ep_r.append(ep_r)


    plt.plot(np.arange(len(all_ep_r)), all_ep_r)
    plt.savefig('./ppo_single2_test.png')
