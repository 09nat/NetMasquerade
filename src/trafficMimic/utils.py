import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../PacketProcessing')))
import yaml
from types import SimpleNamespace
import torch
import torch.nn as nn
import numpy as np
import pickle
import json
import shutil
import pandas as pd
import math
from torch.optim.optimizer import Optimizer
import random
# from PacketProcessing import Flow

os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3, 4'


def recursive_namespace(data):
    if isinstance(data, dict):
        return SimpleNamespace(**{k: recursive_namespace(v) for k, v in data.items()})
    return data


def read_yaml(yaml_pth):
    with open(yaml_pth, 'r') as f:
        content = yaml.safe_load(f)
    return content


def set_device(device_num):
    # print(os.environ['CUDA_VISIBLE_DEVICES'])
    device = torch.device('cuda:{}'.format(device_num) if torch.cuda.is_available() and device_num > -1 else 'cpu')
    return device


class AdamW(Optimizer):
    """Implements Adam algorithm.
    It has been proposed in `Adam: A Method for Stochastic Optimization`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(AdamW, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdamW, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # if group['weight_decay'] != 0:
                #     grad = grad.add(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                # p.data.addcdiv_(-step_size, exp_avg, denom)
                p.data.add_(-step_size, torch.mul(p.data, group['weight_decay']).addcdiv_(1, exp_avg, denom))

        return loss

class ScheduledOptim():
    '''A simple wrapper class for learning rate scheduling'''

    def __init__(self, optimizer, d_model, n_warmup_steps):
        self._optimizer = optimizer
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0
        self.init_lr = np.power(d_model, -0.5)

    def step_and_update_lr(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        "Zero out the gradients by the inner optimizer"
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        return np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])

    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''

        self.n_current_steps += 1
        lr = self.init_lr * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr


def read_flow_pkl(pth):
    flow_list = []
    if isinstance(pth, str):
        if pth.endswith('.pkl'):
            pth = [pth]
        else:
            try:
                pth = [os.path.join(root, file) for root, _, files in os.walk(pth) for file in files if
                       file.endswith('.pkl')]
            except RuntimeError:
                print('fail when read pcap file!')
    for file in pth:
        flow_data = pd.read_pickle(file)
        flow_list += flow_data
    return flow_list


def feature_extract(flow_list):
    '''
    input:
        flow_list: list of flow object, get from read_flow_pkl()
    return:
        timestamp: list of list [[flow1.packet1.timestamp, flow1.packet2.timestamp], [flow2.packet1.timestamp, flow2.packet2.timestamp], ...]
        size: list of list [[flow1.packet1.size, flow1.packet2.size], [flow2.packet1.size, flow2.packet2.size], ...]
    '''
    # timestamp, size: list of list [[flow1.packet1, flow1.packet2], [flow2.packet1, flow2.packet2], ...]
    timestamp = [flow.timestp for flow in flow_list if len(flow.timestp) > 1] # del 1-packet flow
    for stp in timestamp:  # isolation
        for i in range(len(stp) - 1, 0, -1):
            stp[i] = stp[i] - stp[i - 1]
        stp[0] = 0
    size = [flow.total_len for flow in flow_list if len(flow.total_len) > 1]
    return timestamp, size


def feature_extract_full(flow_list):
    timestamp = [flow.timestp for flow in flow_list if len(flow.timestp) > 1] # del 1-packet flow
    key = [flow.key for flow in flow_list if len(flow.timestp) > 1]
    direction = [flow.direction for flow in flow_list if len(flow.timestp) > 1]

    for stp in timestamp:  # isolation
        for i in range(len(stp) - 1, 0, -1):
            stp[i] = stp[i] - stp[i - 1]
        stp[0] = 0
    size = [flow.total_len for flow in flow_list if len(flow.total_len) > 1]
    
    if '/' not in key[0]:
        key = [ele.split('-') for ele in key]
        src_ip = [[key[i][0] if ele == 1 else key[i][1] for ele in direction[i]] for i in range(len(direction))]
        dst_ip = [[key[i][1] if ele == 1 else key[i][0] for ele in direction[i]] for i in range(len(direction))]
        src_port = [[key[i][2] if ele == 1 else key[i][3] for ele in direction[i]] for i in range(len(direction))]
        dst_port = [[key[i][3] if ele == 1 else key[i][2] for ele in direction[i]] for i in range(len(direction))]
        
    else:
        print('Flow identified by Source IP!')
        real_key, others = [], []
        src_ip, dst_ip, src_port, dst_port = [], [], [], []
        for i, item in enumerate(key):
            r, o = item.split('/')
            real_key.append(r)
            others.append(o)
            if isinstance(real_key[0], str):
                src_ip.append([r for _ in range(len(timestamp[i]))])
                # print(others[i])
                dst, s_port, d_port, protocol = others[i].split('-')
                dst_ip.append([dst for _ in range(len(timestamp))])
                src_port.append([s_port for _ in range(len(timestamp))])
                dst_port.append([d_port for _ in range(len(timestamp))])
        
    # time, size, src ip, dst ip, src port, dst port
    return timestamp, size, src_ip, dst_ip, src_port, dst_port


def feature_extract_full_wo_diff(flow_list):
    timestamp = [flow.timestp for flow in flow_list if len(flow.timestp) > 1] # del 1-packet flow
    key = [flow.key for flow in flow_list if len(flow.timestp) > 1]
    direction = [flow.direction for flow in flow_list if len(flow.timestp) > 1]
    size = [flow.total_len for flow in flow_list if len(flow.total_len) > 1]    
    if '/' not in key[0]:
        key = [ele.split('-') for ele in key]
        src_ip = [[key[i][0] if ele == 1 else key[i][1] for ele in direction[i]] for i in range(len(direction))]
        dst_ip = [[key[i][1] if ele == 1 else key[i][0] for ele in direction[i]] for i in range(len(direction))]
        src_port = [[key[i][2] if ele == 1 else key[i][3] for ele in direction[i]] for i in range(len(direction))]
        dst_port = [[key[i][3] if ele == 1 else key[i][2] for ele in direction[i]] for i in range(len(direction))]
        
    else:
        print('Flow identified by Source IP!')
        real_key, others = [], []
        src_ip, dst_ip, src_port, dst_port = [], [], [], []
        for i, item in enumerate(key):
            r, o = item.split('/')
            real_key.append(r)
            others.append(o)
            if isinstance(real_key[0], str):
                src_ip.append([r for _ in range(len(timestamp[i]))])
                dst, s_port, d_port, protocol = others[i].split('-')
                dst_ip.append([dst for _ in range(len(timestamp[i]))])
                src_port.append([s_port for _ in range(len(timestamp[i]))])
                dst_port.append([d_port for _ in range(len(timestamp[i]))])

    return timestamp, size, src_ip, dst_ip, src_port, dst_port


def data_transfer(pth):
    flow = read_flow_pkl(pth)
    random.shuffle(flow)
    timestamp, size = feature_extract(flow)
    concat_list = []
    for i in range(len(timestamp)):
        concat_list.append(np.array(list(zip(size[i], timestamp[i]))))  
    concat_list = np.array(concat_list)
    return concat_list

def data_transfer_save(data, label):
    train_percentage = 0.3
    train_num = int(len(data) * train_percentage)
    valid_percentage = 0.1
    valid_num = int(len(data) * valid_percentage)
    test_percentage = 0.1
    test_num = int(len(data) * test_percentage)

if __name__ == '__main__':
    args = recursive_namespace(read_yaml('/home/lzx/NetMasquerade/Pretrain/config/bert.yaml'))

    # flow_list = read_flow_pkl('/mnt/data/lzx/traffic_data/HVData/storm/flow/storm_m.pkl')

    # print('len flow:', len(flow_list))
    
    # random.shuffle(flow_list)
    # with open('/mnt/data/lzx/traffic_data/HVData/spam1/flow/spam1_m_RL.pkl', 'wb') as f:
    #     pickle.dump(flow_list[:100], f)
    # with open('/mnt/data/lzx/traffic_data/KitSuneData/OS_Scan/SourceIP_flow/Zeus_pkl_malicious_small.pcap.pkl', 'wb') as f:
    #     pickle.dump(flow_list[:100], f)
    # timestamp, size, src_ip, dst_ip, src_port, dst_port = feature_extract_full_wo_diff(flow_list)
    # print(len(size))
    # print(len(timestamp))   
    # print(len(timestamp[5]))
    # for item in size[:5]:
    #     print(item)
        
    # timestamp, size, src_ip, dst_ip, src_port, dst_port = feature_extract_full(flow_list)
    # print(timestamp[0])
    # print(size[0])
    # print(src_ip[0])
    
    
    # 改路径，改save的命名，改大小
    
    
    benign_data_pth = '/mnt/data/lzx/traffic_data/mawi/flow/202306111400_training_demo.pkl'
    malicious_data_pth = '/mnt/data/lzx/traffic_data/Botnets/Waledac_flow/Waledac.br0.1.anonymous.pcap.pkl'
    
    # for Amoeba
    
    benign_concat = data_transfer(benign_data_pth)
    malicious_concat = data_transfer(malicious_data_pth)
    print('benign concat shape: ', benign_concat.shape, benign_concat[0].shape)
    print('malicious concat shape: ', malicious_concat.shape, malicious_concat[0].shape)
    malicious_concat = malicious_concat[:1500]
    benign_concat = benign_concat[:1500]
    malicious_agent = malicious_concat[int(len(malicious_concat) * 0.7):]
    print('malicious: ', malicious_agent.shape)
    np.save('/home/lzx/Amoeba/data/outlier/waledac_agent_train_X.npy', malicious_agent)
    # benign_concat = benign_concat[:1500]
    malicious_concat = malicious_concat[:int(len(malicious_concat) * 0.7)]
    
    label = [1 for _ in range(len(benign_concat))] + [0 for _ in range(len(malicious_concat))]
    label = np.array(label)
    data = np.concatenate((benign_concat, malicious_concat), axis=0)
    assert label.shape[0] == data.shape[0]
    shuffle_indices = np.random.permutation(len(data))
    data = data[shuffle_indices]
    label = label[shuffle_indices]
    print('label shape: ', data.shape, label.shape)
    
    np.save('/home/lzx/Amoeba/data/outlier/waledac_train_X.npy', data[:int(len(data) * 0.3)])
    np.save('/home/lzx/Amoeba/data/outlier/waledac_train_y.npy', label[:int(len(data) * 0.3)])
    np.save('/home/lzx/Amoeba/data/outlier/waledac_valid_X.npy', data[int(len(data) * 0.3):int(len(data) * 0.4)])
    np.save('/home/lzx/Amoeba/data/outlier/waledac_valid_y.npy', label[int(len(data) * 0.3):int(len(data) * 0.4)])
    
    
    test_index = int(len(data) * 0.4)
    filtered_data = data[test_index:]
    filtered_label = label[test_index:]
    
    filtered_data = filtered_data[filtered_label == 0]
    filtered_label = filtered_label[filtered_label == 0]
    
    print('filter:', filtered_data.shape, filtered_label.shape)
    np.save('/home/lzx/Amoeba/data/outlier/waledac_test_X.npy', filtered_data)
    np.save('/home/lzx/Amoeba/data/outlier/waledac_test_y.npy', filtered_label)
    print('waledac dataset saved')    
    
    
    

    