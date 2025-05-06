# This file is a simulation version of NetMasquerade (with rl_dateset).

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import math
import numpy as np
import torch
import torch.nn as nn
import bisect
from copy import deepcopy
from torch.utils.data import Dataset, DataLoader
from trafficMimic.utils import * 
from trafficMimic.dataset.vocab import TimeVocab, SizeVocab
from advGenerate.rl_dataset import RLFlowDataset, RLFullFlowDataset, RLFullFlowNoDiffDataset
from trafficMimic.model.architecture import BERT, BERTLM
from rl_utils import *
from advGenerate.environments.KitSune import KitSune_Evaluate
from advGenerate.environments.LSTM import LSTM_Evaluate
from advGenerate.environments.NetBeacon import NetBeacon_Evaluate
from advGenerate.environments.Flowlens import Flowlens_Evaluate
from advGenerate.environments.Whisper import Whisper_Evaluate
from advGenerate.environments.MLP import MLP_Evaluate


class BaseEnv(object):
    def __init__(self, rl_args, bert_args, mode='train'):
        self.rl_args = rl_args
        self.bert_args = bert_args
        self.mode = mode
        self.device = set_device(self.rl_args.trainer.device)

        self.timevocab = TimeVocab.load_vocab(self.bert_args.trainer.timevocab_pth)
        self.sizevocab = SizeVocab.load_vocab(self.bert_args.trainer.sizevocab_pth)

        # self.train_dataset = RLFullFlowDataset(self.rl_args, self.timevocab, self.sizevocab, mode)
        self.train_dataset = RLFullFlowNoDiffDataset(self.rl_args, self.timevocab, self.sizevocab, mode)    # for Kitsune
        
        self.data_iter = 0
        print('the length of RL dataset: ', len(self.train_dataset))

        self.bert = BERT(len(self.timevocab), len(self.sizevocab), self.bert_args.model.hidden, self.bert_args.model.d_ff,
                self.bert_args.model.n_layers, self.bert_args.model.attn_heads, self.bert_args.model.dropout)
        self.model = BERTLM(self.bert, len(self.timevocab), len(self.sizevocab)).to(self.device)
        self.model.load_state_dict(torch.load(self.rl_args.trainer.bert_pth, map_location=self.device))
        self.model.eval()
        self.state = None
        self.step_num = 0
        self.original_index = []
        self.src = []
        self.dst = []
        self.srcport = []
        self.dstport = []
        self.src_index = []
        self.flow = None
         
    def reset(self, ): 
        self.step_num = 0
        initial_state, self.real_feat, self.src, self.dst, self.srcport, self.dstport, self.flow = deepcopy(self.train_dataset[self.data_iter])
        self.data_iter = self.data_iter + 1 if self.data_iter < len(self.train_dataset) - 1 else 0

        self.state = {k: v.to(self.device) for k, v in initial_state.items()}
        
        if self.state['real_length'].item() > self.rl_args.model.state_dim:
            self.state['real_length'] = torch.tensor(self.rl_args.model.state_dim).to(self.device)

        self.original_index = list(range(1, self.state['real_length'].item() - 1))
        if self.rl_args.env.bad_ip is None:
            self.src_index = [i + 1 for i, x in enumerate(self.src) if x == self.src[0] and i < 510]
        else:
            self.src_index = [i + 1 for i, x in enumerate(self.src) if x == self.rl_args.env.bad_ip and i < 510]
    
        fill_ones = torch.zeros_like(self.state['flow_ipd'], device=self.device)
        fill_ones[self.src_index] = 1
        self.state['src_index'] = fill_ones
        
    def step(self, action):
        pass

    def tran_state(self, action, ipd_window=None, size_window=None):
        mask_ipd = self.state['flow_ipd'].clone()
        mask_size = self.state['flow_size'].clone()

        if action % 2 == 0: # insert
            mask_index = action // 2

            mask_ipd = self.insert(mask_ipd, self.timevocab.mask_index, mask_index)
            mask_size = self.insert(mask_size, self.sizevocab.mask_index, mask_index)

            if self.state['real_length'].item() < self.rl_args.model.state_dim:
                self.state['real_length'] += 1
                
            # modify original_index
            for i in range(len(self.original_index)):
                if self.original_index[i] >= mask_index:
                    self.original_index[i] += 1
            if self.original_index[-1] >= self.rl_args.model.state_dim:
                self.original_index = self.original_index[:-1]

            # modify address and port info.
            for i in range(len(self.src_index)):
                if self.src_index[i] >= mask_index:
                    self.src_index[i] += 1
            bisect.insort(self.src_index, mask_index)
            if self.src_index[-1] >= 511:
                self.src_index = self.src_index[:-1]
            
            if self.rl_args.env.bad_ip is None:
                self.src.insert(mask_index - 1, self.src[0])
                self.srcport.insert(mask_index - 1, self.srcport[0])
                self.dst.insert(mask_index - 1, self.dst[0])
                self.dstport.insert(mask_index - 1, self.dstport[0])
            else:
                bad_pos = 0
                for i, v in enumerate(self.src):
                    if v == self.rl_args.env.bad_ip:
                        bad_pos = i
                        break
                self.src.insert(mask_index - 1, self.rl_args.env.bad_ip)
                self.dst.insert(mask_index - 1, self.dst[bad_pos])
                self.srcport.insert(mask_index - 1, self.srcport[bad_pos])
                self.dstport.insert(mask_index - 1, self.dstport[bad_pos])
                
            # modify state
            ipd_pred, size_pred = self.model(mask_ipd.unsqueeze(0), mask_size.unsqueeze(0))
            
            if ipd_window is None:
                ipd_pred = ipd_pred[:, :, 5:].argmax(dim=-1) + 5
            else:
                ipd_pred = ipd_pred[:, :, ipd_window[0]: ipd_window[1]].argmax(dim=-1) + ipd_window[0]
            
            
            if size_window is None:
                size_pred = size_pred[:, :, 25:].argmax(dim=-1) + 25
            else:
                size_pred = size_pred[:, :, size_window[0]: size_window[1]].argmax(dim=-1) + size_window[0]

            mask_ipd[mask_index] = ipd_pred[0, mask_index]
            mask_size[mask_index] = size_pred[0, mask_index]
  
            self.state['flow_ipd'] = mask_ipd
            self.state['flow_size'] = mask_size

            fill_ones = torch.zeros_like(mask_ipd, device=self.device)

            fill_ones[self.src_index] = 1
            self.state['src_index'] = fill_ones
            
            # modify real
            self.real_feat['ipd'].insert(mask_index - 1, self.timevocab.itos(mask_ipd[mask_index].item()))
            self.real_feat['size'].insert(mask_index - 1, self.sizevocab.itos(mask_size[mask_index].item()))
            
            if len(self.real_feat['ipd']) > self.rl_args.model.state_dim - 1:
                self.real_feat['ipd'] = self.real_feat['ipd'][:self.rl_args.model.state_dim - 1]
                self.real_feat['size'] = self.real_feat['size'][:self.rl_args.model.state_dim - 1]


        else: # modify
            standard_size = deepcopy(mask_size)
            mask_index = action // 2

            mask_ipd[mask_index] = self.timevocab.mask_index
            mask_size[mask_index] = self.sizevocab.mask_index

            ipd_pred, size_pred = self.model(mask_ipd.unsqueeze(0), mask_size.unsqueeze(0))
            
            
            if ipd_window is None:
                ipd_pred = ipd_pred[:, :, 5:].argmax(dim=-1) + 5
            else:
                ipd_pred = ipd_pred[:, :, ipd_window[0]: ipd_window[1]].argmax(dim=-1) + ipd_window[0]

            mask_ipd[mask_index] = ipd_pred[0, mask_index]
            
            self.state['flow_ipd'] = mask_ipd
            self.state['flow_size'] = standard_size
            self.real_feat['ipd'][mask_index - 1] = self.timevocab.itos(mask_ipd[mask_index].item())

    def insert(self, seq, val, pos):
        left = seq[:pos]
        right = seq[pos:]
        tensor_inserted = torch.cat([left, torch.tensor([val], device=seq.device), right])
        return tensor_inserted[:seq.shape[0]]


class KitSuneEnv(BaseEnv):
    def __init__(self, rl_args, bert_args, mode='train'):
        super().__init__(rl_args, bert_args, mode)
        self.eval = KitSune_Evaluate(self.rl_args.trainer.target_model_pth, self.rl_args.trainer.device)
        self.acc = 0
        self.adv_flow = []

    def reset(self):
        super().reset()
        self.acc = self.get_acc()
        print('init acc:', self.acc)
        return self.state
    
    def get_acc(self,):
        ipd_list = deepcopy(self.real_feat['ipd'])
        
        init_timestamp = self.flow.timestp[0]
        size_list = self.real_feat['size']
        
        for i in range(len(ipd_list)):
            init_timestamp += ipd_list[i]
            ipd_list[i] = init_timestamp
        
        data = self.eval.get_data(ipd_list, size_list, self.src, self.dst, self.srcport, self.dstport, self.original_index)
        result = self.eval.evaluate(data)

        result = [result[i - 1] for i in self.src_index]
        if len(result) == 0:
            return -1
        acc = 1.0 * sum(result) / len(result)
        return acc  

    def step(self, action):
        self.tran_state(action, ipd_window=[10, 40], size_window=[100, 400])

        done = False
        acc = self.get_acc()

        discrepancy = acc - self.acc
        self.acc = acc
        self.step_num += 1
        
        if self.step_num >= self.rl_args.trainer.max_stop_step + 1:
            done = True
        
        if self.acc < 0.03:
            done = True
        
        return self.state, - discrepancy, done, 1 - self.acc
    
    def save_state(self, ):
        init_timestamp = self.flow.timestp[0]
        ipd_list = deepcopy(self.real_feat['ipd'])
        for i in range(len(ipd_list)):
            init_timestamp += ipd_list[i]
            ipd_list[i] = init_timestamp
            
        ipd = [ipd_list[i - 1] for i in self.src_index]
        self.adv_flow.append(ipd)
    

class LSTMEnv(BaseEnv):
    def __init__(self, rl_args, bert_args, mode='train'):
        super().__init__(rl_args, bert_args, mode)
        self.eval = LSTM_Evaluate(self.rl_args.trainer.target_model_pth, self.rl_args.trainer.device)
        self.total_step = -1

    def reset(self):
        super().reset()
        self.get_res()
        self.total_step = -1
        return self.state
    
    def get_res(self,):

        ipd_list = self.real_feat['ipd']
        size_list = self.real_feat['size']

        ipd_tensor = torch.tensor(ipd_list).to(self.device)
        size_tensor = torch.tensor(size_list).to(self.device)
        
        res = self.eval.evaluate(ipd_tensor, size_tensor)
        
        return res

    def step(self, action):
        self.total_step += 1
        self.tran_state(action)

        result = self.get_res()
        done = False
        reward = 0
        self.step_num += 1

        if self.step_num >= self.rl_args.trainer.max_stop_step + 1:
            done = True
            reward = 0
        
        if result == 0:
            done = True
            reward = 1
            
        step_penalty = 0 if reward == 1 else -0.03
        
        return self.state, reward + step_penalty, done, 1 - result
    
    def constraint_penalty(self, ):
        ipd_list = self.real_feat['ipd']
        total_time = sum(ipd_list[self.original_index[0] + 1: self.original_index[-1] + 1])
        
    def insert(self, seq, val, pos):
        left = seq[:pos]
        right = seq[pos:]
        tensor_inserted = torch.cat([left, torch.tensor([val], device=seq.device), right])
        return tensor_inserted[:seq.shape[0]]        
        
   
   
class NetBeaconEnv(BaseEnv):
    def __init__(self, rl_args, bert_args, mode='train'):
        super().__init__(rl_args, bert_args, mode)
        self.eval = NetBeacon_Evaluate(self.rl_args.trainer.target_model_pth, self.rl_args.trainer.device)
        self.total_step = -1

    def reset(self):
        super().reset()
        return self.state
    
    def get_res(self,):
        ipd_list = self.real_feat['ipd']
        size_list = self.real_feat['size']
        res = self.eval.evaluate([ipd_list, size_list])
        return res

    def step(self, action):
        self.total_step += 1
        self.get_res()
        self.tran_state(action)

        result = self.get_res()
        done = False
        reward = 0
        self.step_num += 1

        if self.step_num >= self.rl_args.trainer.max_stop_step + 1:
            done = True
            reward = 0
        
        if result == 0:
            done = True
            reward = 1
            
        step_penalty = 0 if reward == 1 else -0.03
            
        return self.state, reward + step_penalty, done, 1 - result
    
    
class FlowlensEnv(BaseEnv):
    def __init__(self, rl_args, bert_args, mode='train'):
        super().__init__(rl_args, bert_args, mode)
        self.eval = Flowlens_Evaluate(self.rl_args.trainer.target_model_pth, self.rl_args.trainer.device)

    def reset(self):
        super().reset()
        return self.state
    
    def get_res(self,):
        real_length = self.state['real_length'].item() 

        ipd_list = self.real_feat['ipd']
        size_list = self.real_feat['size']
        return self.eval.evaluate((ipd_list, size_list))

    def step(self, action):
        self.tran_state(action)

        result = self.get_res()
        done = False
        reward = 0
        self.step_num += 1

        if self.step_num >= self.rl_args.trainer.max_stop_step + 1:
            done = True
            reward = 0
        
        if result == 0:
            done = True
            reward = 1
            
        step_penalty = 0 if reward == 1 else -0.01    

        return self.state, reward + step_penalty, done, 1 - result


class WhisperEnv(BaseEnv):
    def __init__(self, rl_args, bert_args, mode='train'):
        super().__init__(rl_args, bert_args, mode)
        self.eval = Whisper_Evaluate(self.rl_args.trainer.target_model_pth, self.rl_args.trainer.device)

    def reset(self):
        super().reset()
        print('new', end=' ')
        self.get_res()
        return self.state
    
    def get_res(self,):
        ipd_list = self.real_feat['ipd']
        size_list = self.real_feat['size']
        ipd_tensor = torch.tensor(ipd_list).to(self.device)
        size_tensor = torch.tensor(size_list).to(self.device)

        res = self.eval.evaluate([ipd_tensor, size_tensor])
        print(res)
        
        return res

    def step(self, action):
        self.tran_state(action)

        result = self.get_res()
        done = False
        reward = 0
        self.step_num += 1

        if self.step_num >= self.rl_args.trainer.max_stop_step + 1:
            done = True
            reward = 0
        
        if result == 0:
            done = True
            reward = 1
        return self.state, reward, done, 1 - result
    

class MLPEnv(BaseEnv):
    def __init__(self, rl_args, bert_args, mode='train'):
        super().__init__(rl_args, bert_args, mode)
        self.eval = MLP_Evaluate(self.rl_args.trainer.target_model_pth, self.rl_args.trainer.device)
        self.latest = 0

    def reset(self):
        super().reset()
        self.get_res()
        self.latest = self.constraint_penalty()
        return self.state
    
    def get_res(self,):
        result = self.eval.evaluate([self.real_feat['ipd'], self.real_feat['size'], self.src, self.dst, self.srcport, self.dstport])
        return result

    def step(self, action):
        self.tran_state(action)

        result = self.get_res()
        done = False
        reward = 0
        self.step_num += 1

        if self.step_num >= self.rl_args.trainer.max_stop_step + 1:
            done = True
            reward = 0
        
        
        if result == 0:
            done = True
            reward = 1
        
        latest = self.constraint_penalty()
        dos_penalty = self.latest - latest
        self.latest = latest
        
        step_penalty = 0 if reward == 1 else -0.01

        return self.state, reward + step_penalty + 0. * dos_penalty, done, 1 - result

    def constraint_penalty(self, ):
        total_time = sum(self.real_feat['ipd'][self.original_index[0]: self.original_index[-1] - 1])
        
        return total_time
    

class TestEnv(BaseEnv):
    def __init__(self, rl_args, bert_args):
        super().__init__(rl_args, bert_args)

    def reset(self):
        super().reset()
        return self.state
    
    def step(self, action):
        real_length = self.state['real_length'].item()
        if action <= 1 or action >= 2 * real_length - 1:
            return self.state, -5, False
        
        else:            
            self.tran_state(action)
        
            reward = 1
            done = False
            self.obs_state()
            return self.state, reward, done
        
    def obs_state(self, ):
        print('flow_ipd: ', self.state['flow_ipd'][0, :self.state['real_length'].data])
        print('flow_size: ', self.state['flow_size'][0, :self.state['real_length'].data])


if __name__ == '__main__':
    pass
 
