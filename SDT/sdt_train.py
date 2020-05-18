# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter

def onehot_coding(target, device, output_dim):
    target_onehot = torch.FloatTensor(target.size()[0], output_dim).to(device)
    target_onehot.data.zero_()
    target_onehot.scatter_(1, target.view(-1, 1), 1.)
    return target_onehot
use_cuda = True
learner_args = {'input_dim': 8,
                'output_dim': 4,
                'depth': 3,
                'lamda': 1e-3,
                'lr': 1e-3,
                'weight_decay': 0.,  # 5e-4
                'batch_size': 1280,
                'epochs': 40,
                'cuda': use_cuda,
                'log_interval': 100,
                'exp_scheduler_gamma': 1.,
                'beta' : True,  # temperature 
                'greatest_path_probability': True  # when forwarding the SDT, \
                # choose the leaf with greatest path probability or average over distributions of all leaves; \
                # the former one has better explainability while the latter one achieves higher accuracy
                }
# learner_args['model_path'] = './model/sdt_'+str(learner_args['depth'])
learner_args['model_path'] = './model/trees/sdt_'+str(learner_args['depth'])

device = torch.device('cuda' if use_cuda else 'cpu')
