# -*- coding: utf-8 -*-
'''' Soft Decision Tree '''
import torch
import torch.nn as nn
from collections import OrderedDict


class SDT(nn.Module):
    """ Soft Desicion Tree """
    def __init__(self, args):
        super(SDT, self).__init__()
        self.args = args
        self.device = torch.device('cuda' if self.args['cuda'] else 'cpu')
        self.inner_node_num = 2 ** self.args['depth'] - 1
        self.leaf_num = 2 ** self.args['depth']
        self.max_depth = self.args['depth']
        self.max_leaf_idx=None  # the leaf index with maximal path probability
        
        # Different penalty coefficients for nodes in different layer
        self.penalty_list = [args['lamda'] * (2 ** (-depth)) for depth in range(0, self.args['depth'])] 
        
        # inner nodes operation
        # Initialize inner nodes and leaf nodes (input dimension on innner nodes is added by 1, serving as bias)
        self.linear = nn.Linear(self.args['input_dim']+1, self.inner_node_num, bias=False)
        self.sigmoid = nn.Sigmoid()
        # temperature term
        if self.args['beta']:
            beta = torch.randn(self.inner_node_num)
            self.beta = nn.Parameter(beta)
        else:
            self.beta = torch.ones(1).to(self.device)   # or use one beta across all nodes

        # leaf nodes operation
        # p*softmax(Q) instead of softmax(p*Q)
        param = torch.randn(self.leaf_num, self.args['output_dim'])
        self.param = nn.Parameter(param)
        self.softmax = nn.Softmax(dim=1)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.args['lr'], weight_decay=self.args['weight_decay'])
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.args['exp_scheduler_gamma'])

    def leaf_nodes(self, p):
        distribution_per_leaf = self.softmax(self.param)
        average_distribution = torch.mm(p, distribution_per_leaf)
        return average_distribution

    def inner_nodes(self, x):
        output = self.sigmoid(self.beta*self.linear(x))
        return output

    def get_tree_weights(self, Bias=False):
        """Return tree weights as a list"""
        if Bias:
            return self.state_dict()['linear.weight'].detach().cpu().numpy()   
        else:  # no bias
            return self.state_dict()['linear.weight'][:, 1:].detach().cpu().numpy()


    def forward(self, data, LogProb=True):
        _mu, _penalty = self._forward(data)
        output = self.leaf_nodes(_mu) # average over leaves

        if self.args['greatest_path_probability']:
            one_hot_path_probability = torch.zeros(_mu.shape).to(self.device)
            vs, ids = torch.max(_mu, 1)  # ids is the leaf index with maximal path probability
            one_hot_path_probability.scatter_(1, ids.view(-1,1), 1.)
 
            prediction = self.leaf_nodes(one_hot_path_probability)
            self.max_leaf_idx = ids

        else:  # prediction value equals to the average distribution
            prediction = output

        if LogProb:
            output = torch.log(output)
            prediction = torch.log(prediction)

        weights = self.get_tree_weights(Bias=True)

        return prediction, output, _penalty, weights
    
    """ Core implementation on data forwarding in SDT """
    def _forward(self, data):
        batch_size = data.size()[0]
        data = self._data_augment_(data)
        path_prob = self.inner_nodes(data)
        path_prob = torch.unsqueeze(path_prob, dim=2)
        path_prob = torch.cat((path_prob, 1-path_prob), dim=2)
        _mu = data.data.new(batch_size,1,1).fill_(1.)
        _penalty = torch.tensor(0.).to(self.device)
        
        begin_idx = 0
        end_idx = 1
        
        for layer_idx in range(0, self.args['depth']):
            _path_prob = path_prob[:, begin_idx:end_idx, :]
            _penalty= _penalty + self._cal_penalty(layer_idx, _mu, _path_prob)  # extract inner nodes in current layer to calculate regularization term
            _mu = _mu.view(batch_size, -1, 1).repeat(1, 1, 2)
            _mu = _mu * _path_prob
            begin_idx = end_idx  # index for each layer
            end_idx = begin_idx + 2 ** (layer_idx+1)
        mu = _mu.view(batch_size, self.leaf_num)            

        return mu, _penalty   # mu contains the path probability for each leaf       
    
    """ Calculate penalty term for inner-nodes in different layer """
    def _cal_penalty(self, layer_idx, _mu, _path_prob):
        penalty = torch.tensor(0.).to(self.device)     
        batch_size = _mu.size()[0]
        _mu = _mu.view(batch_size, 2**layer_idx)
        _path_prob = _path_prob.view(batch_size, 2**(layer_idx+1))
        for node in range(0, 2**(layer_idx+1)):
            numerical_bound = 1e-7  # prevent numerical issue
            alpha = torch.sum(_path_prob[:, node]*_mu[:,node//2], dim=0) / (torch.sum(_mu[:,node//2], dim=0) + numerical_bound)  # not dividing 0.
            origin_alpha=alpha
            # if alpha ==1 or alpha ==  0, log will cause numerical problem, so alpha should be bounded
            alpha = torch.clamp(alpha, numerical_bound, 1-numerical_bound)  # no log(negative value)
            alpha_list.append(alpha)
            if torch.isnan(torch.tensor(alpha_list)).any():
                print(origin_alpha, alpha)
                
            penalty -= self.penalty_list[layer_idx] * 0.5 * (torch.log(alpha) + torch.log(1-alpha))
        return penalty
    
    """ Add constant 1 onto the front of each instance, serving as the bias """
    def _data_augment_(self, input):
        batch_size = input.size()[0]
        input = input.view(batch_size, -1)
        bias = torch.ones(batch_size, 1).to(self.device)
        input = torch.cat((bias, input), 1)
        return input

    def save_model(self, model_path, id=''):
        torch.save(self.state_dict(), model_path+id)

    def load_model(self, model_path, id=''):
        self.load_state_dict(torch.load(model_path+id, map_location='cpu'))
        self.eval()

