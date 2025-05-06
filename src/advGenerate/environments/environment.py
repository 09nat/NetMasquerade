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
from advGenerate.environments.SVM import SVM_Evaluate
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
        # self.dataloader = DataLoader(self.train_dataset, batch_size=1, shuffle=False, drop_last=True)
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
        
        self.sock = socket_conn("103.233.162.230", 61616)
        
    def reset(self, ): 
        self.step_num = 0
        initial_state, self.real_feat, self.src, self.dst, self.srcport, self.dstport, self.flow = deepcopy(self.train_dataset[self.data_iter])
        # print(type(self.flow))
        self.data_iter = self.data_iter + 1 if self.data_iter < len(self.train_dataset) - 1 else 0
        # self.data_iter = self.data_iter + 1 if self.data_iter < 2 else 0

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
            # print('ipd_pred.shape', ipd_pred.shape)
            
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
            
            # '''
            # attention!
            # '''
            # if ipd_window is not None:
            #     mask_ipd[mask_index] += 40
            #     if mask_ipd[mask_index].item() > ipd_window:
            #         mask_ipd[mask_index] = ipd_window
                
            # print(ipd_pred[0, mask_index].item())
            # print(self.timevocab.itos(mask_ipd[mask_index].item()))
        
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
            
            # '''
            # attention!
            # '''
            # if ipd_window > 0:
            #     mask_ipd[mask_index] += 40
            #     if mask_ipd[mask_index].item() > ipd_window:
            #         mask_ipd[mask_index] = ipd_window
                
            
            self.state['flow_ipd'] = mask_ipd
            self.state['flow_size'] = standard_size
            self.real_feat['ipd'][mask_index - 1] = self.timevocab.itos(mask_ipd[mask_index].item())

    def insert(self, seq, val, pos):
        left = seq[:pos]
        right = seq[pos:]
        tensor_inserted = torch.cat([left, torch.tensor([val], device=seq.device), right])
        # print("Inserted tensor:", tensor_inserted[:, 1: self.state['real_length'].item() + 1])
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
        # real_length = self.state['real_length'].item() 
        # ipd_list = self.timevocab.from_seq(self.state['flow_ipd'].squeeze().tolist()[1: real_length - 1])
        # size_list = self.sizevocab.from_seq(self.state['flow_size'].squeeze().tolist()[1: real_length - 1])
        ipd_list = deepcopy(self.real_feat['ipd'])
        
        # print('ipd list before:', ipd_list)
        init_timestamp = self.flow.timestp[0]
        # print('init_timestamp', init_timestamp)
        size_list = self.real_feat['size']
        
        for i in range(len(ipd_list)):
            init_timestamp += ipd_list[i]
            ipd_list[i] = init_timestamp
        
        # print('ipd list now:', ipd_list)

        data = self.eval.get_data(ipd_list, size_list, self.src, self.dst, self.srcport, self.dstport, self.original_index)
        result = self.eval.evaluate(data)

        # print(len(ipd_list))
        # print(self.original_index)

        # print(len(result), len(self.src_index))
        # print(self.src_index)
        # print(self.src_index)
        result = [result[i - 1] for i in self.src_index]
        # print(result)
        if len(result) == 0:
            return -1
        acc = 1.0 * sum(result) / len(result)
        # print(acc)
        return acc  

    def step(self, action):
        # print('action:', action)
        self.tran_state(action, ipd_window=[10, 40], size_window=[100, 400])
        # self.tran_state(action)
        
        # print('ipd:', self.real_feat['ipd'])
        # print('size:', self.real_feat['size'])

        done = False
        acc = self.get_acc()
        # print('step acc: ', acc)

        discrepancy = acc - self.acc
        self.acc = acc
        self.step_num += 1
        
        if self.step_num >= self.rl_args.trainer.max_stop_step + 1:
            done = True
        
        if self.acc < 0.03:
            done = True
        # print(self.acc)
        # return self.state, 1 - self.acc, done, 1 - self.acc
        
        # if done:
            # print('done, ', self.acc, self.step_num)
            # self.save_state()
        
        return self.state, - discrepancy, done, 1 - self.acc
    
    def save_state(self, ):
        init_timestamp = self.flow.timestp[0]
        # print('init_timestamp', init_timestamp)
        ipd_list = deepcopy(self.real_feat['ipd'])
        # print(ipd_list)
        for i in range(len(ipd_list)):
            init_timestamp += ipd_list[i]
            ipd_list[i] = init_timestamp
            
        ipd = [ipd_list[i - 1] for i in self.src_index]
        self.adv_flow.append(ipd)
    

class LSTMEnv(BaseEnv):
    def __init__(self, rl_args, bert_args, mode='train'):
        super().__init__(rl_args, bert_args, mode)
        self.eval = LSTM_Evaluate(self.rl_args.trainer.target_model_pth, self.rl_args.trainer.device)
        self.totol_step = -1
        # self.acc = 0

    def reset(self):
        super().reset()
        # print('new', end=' ')
        self.get_res()
        self.totol_step = -1
        return self.state
    
    def get_res(self,):
        # real_length = self.state['real_length'].item() 
        # ipd_list = self.timevocab.from_seq(self.state['flow_ipd'].squeeze().tolist()[1: real_length - 1])
        # size_list = self.sizevocab.from_seq(self.state['flow_size'].squeeze().tolist()[1: real_length - 1])

        # print('time: ', ipd_list)
        # print('size: ', size_list)

        ipd_list = self.real_feat['ipd']
        size_list = self.real_feat['size']

        ipd_tensor = torch.tensor(ipd_list).to(self.device)
        size_tensor = torch.tensor(size_list).to(self.device)
        
        res = self.eval.evaluate(ipd_tensor, size_tensor)
        
        # print(self.totol_step, ' state: ', ipd_list[:8], size_list[:8], ' res: ', res)
        return res

    def step(self, action):
        self.totol_step += 1
        # print('action: ', action)
        self.tran_state(action)
        # self.trans_ablation(action, type='t')

        result = self.get_res()
        done = False
        reward = 0
        self.step_num += 1


        if self.step_num >= self.rl_args.trainer.max_stop_step + 1:
            done = True
            reward = 0
        
        # reward = -0.01 * self.step_num
        # if self.step_num > 10 and self.state == old_state:
        #     return self.state, -1, True
        
        if result == 0:
            done = True
            reward = 1
            
        step_penalty = 0 if reward == 1 else -0.03
        
        return self.state, reward + step_penalty, done, 1 - result
    
    def constraint_penalty(self, ):
        # print(self.original_index)
        # ipd_list = self.timevocab.from_seq(self.state['flow_ipd'].squeeze().tolist())
        ipd_list = self.real_feat['ipd']
        total_time = sum(ipd_list[self.original_index[0] + 1: self.original_index[-1] + 1])
        # print(total_time)
        
    def insert(self, seq, val, pos):
        left = seq[:pos]
        right = seq[pos:]
        tensor_inserted = torch.cat([left, torch.tensor([val], device=seq.device), right])
        # print("Inserted tensor:", tensor_inserted[:, 1: self.state['real_length'].item() + 1])
        return tensor_inserted[:seq.shape[0]]        
        
   
    def trans_ablation(self, action, type='s'):
        specified_ipd_token = 0
        specified_ipd_value = 0
        specified_size_token = 0
        specified_size_value = 0
        mean_ipd = sum(self.real_feat['ipd']) / len(self.real_feat['ipd'])
        mean_size = sum(self.real_feat['size']) // len(self.real_feat['size'])
        max_ipd = max(self.real_feat['ipd'])
        max_size = max(self.real_feat['size'])
        min_ipd = min(self.real_feat['ipd'])
        min_size = min(self.real_feat['size'])
        
        if type == 's':
            specified_ipd_value = random.uniform(min_ipd, max_ipd)
            specified_size_value = random.randint(min_size, max_size)
            # specified_size_value = random.randint(20, 1500)
            specified_ipd_token = self.timevocab.stoi(specified_ipd_value)
            specified_size_token = specified_size_value + 5
        else:
            specified_ipd_value = mean_ipd
            specified_size_value = mean_size
            specified_ipd_token = self.timevocab.stoi(specified_ipd_value)
            specified_size_token = specified_size_value + 5

        if action % 2 == 0: # insert
            mask_index = action // 2

            self.state['flow_ipd'] = self.insert(self.state['flow_ipd'], specified_ipd_token, mask_index)
            self.state['flow_size'] = self.insert(self.state['flow_size'], specified_size_token, mask_index)

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

            fill_ones = torch.zeros_like(self.state['flow_ipd'], device=self.device)

            fill_ones[self.src_index] = 1
            self.state['src_index'] = fill_ones
            
            # modify real
            self.real_feat['ipd'].insert(mask_index - 1, specified_ipd_value)
            self.real_feat['size'].insert(mask_index - 1, specified_size_value)
            
            if len(self.real_feat['ipd']) > self.rl_args.model.state_dim - 1:
                self.real_feat['ipd'] = self.real_feat['ipd'][:self.rl_args.model.state_dim - 1]
                self.real_feat['size'] = self.real_feat['size'][:self.rl_args.model.state_dim - 1]


        else: # modify
            mask_index = action // 2
            
            self.state['flow_ipd'][mask_index] = specified_ipd_token
            self.real_feat['ipd'][mask_index - 1] = specified_ipd_value
   
   
class NetBeaconEnv(BaseEnv):
    def __init__(self, rl_args, bert_args, mode='train'):
        super().__init__(rl_args, bert_args, mode)
        self.eval = NetBeacon_Evaluate(self.rl_args.trainer.target_model_pth, self.rl_args.trainer.device)
        self.totol_step = -1

    def reset(self):
        super().reset()
        return self.state
    
    def get_res(self,):
        ipd_list = self.real_feat['ipd']
        size_list = self.real_feat['size']
        # print(len(ipd_list))
        res = self.eval.evaluate([ipd_list, size_list])
        # print(self.totol_step, ' state: ', ipd_list, size_list, ' res: ', res)
        return res

    def step(self, action):
        self.totol_step += 1
        self.get_res()
        self.tran_state(action)
        # self.tran_state(action, ipd_window=[30, 40])
        # self.trans_ablation(action, type='f')

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
    
    def trans_ablation(self, action, type='s'):
        specified_ipd_token = 0
        specified_ipd_value = 0
        specified_size_token = 0
        specified_size_value = 0
        mean_ipd = sum(self.real_feat['ipd']) / len(self.real_feat['ipd'])
        mean_size = sum(self.real_feat['size']) // len(self.real_feat['size'])
        max_ipd = max(self.real_feat['ipd'])
        max_size = max(self.real_feat['size'])
        min_ipd = min(self.real_feat['ipd'])
        min_size = min(self.real_feat['size'])
        
        if type == 's':
            specified_ipd_value = 10 ** np.random.uniform(np.log(min_ipd + 1e-9), np.log(max_ipd + 1e-9))
            specified_size_value = random.randint(min_size, max_size)
            # specified_size_value = random.randint(20, 1500)
            specified_ipd_token = self.timevocab.stoi(specified_ipd_value)
            specified_size_token = specified_size_value + 5
        else:
            specified_ipd_value = mean_ipd
            specified_size_value = mean_size
            specified_ipd_token = self.timevocab.stoi(specified_ipd_value)
            specified_size_token = specified_size_value + 5

        if action % 2 == 0: # insert
            mask_index = action // 2

            self.state['flow_ipd'] = self.insert(self.state['flow_ipd'], specified_ipd_token, mask_index)
            self.state['flow_size'] = self.insert(self.state['flow_size'], specified_size_token, mask_index)

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

            fill_ones = torch.zeros_like(self.state['flow_ipd'], device=self.device)

            fill_ones[self.src_index] = 1
            self.state['src_index'] = fill_ones
            
            # modify real
            self.real_feat['ipd'].insert(mask_index - 1, specified_ipd_value)
            self.real_feat['size'].insert(mask_index - 1, specified_size_value)
            
            if len(self.real_feat['ipd']) > self.rl_args.model.state_dim - 1:
                self.real_feat['ipd'] = self.real_feat['ipd'][:self.rl_args.model.state_dim - 1]
                self.real_feat['size'] = self.real_feat['size'][:self.rl_args.model.state_dim - 1]


        else: # modify
            mask_index = action // 2
            
            self.state['flow_ipd'][mask_index] = specified_ipd_token
            self.real_feat['ipd'][mask_index - 1] = specified_ipd_value
    
    
class SVMEnv(BaseEnv):
    def __init__(self, rl_args, bert_args, mode='train'):
        super().__init__(rl_args, bert_args, mode)
        self.eval = SVM_Evaluate(self.rl_args.trainer.target_model_pth, self.rl_args.trainer.device)

    def reset(self):
        super().reset()
        return self.state
    
    def get_res(self,):
        real_length = self.state['real_length'].item() 
        # ipd_list = self.timevocab.from_seq(self.state['flow_ipd'].squeeze().tolist()[1: real_length - 1])
        # size_list = self.sizevocab.from_seq(self.state['flow_size'].squeeze().tolist()[1: real_length - 1])
        
        # print('ipd_list:', ipd_list)
        # print('size_list:', size_list)
        ipd_list = self.real_feat['ipd']
        size_list = self.real_feat['size']
        return self.eval.evaluate((ipd_list, size_list))

    def step(self, action):
        # real_length = self.state['real_length'].item()
        # ipd_list = self.timevocab.from_seq(self.state['flow_ipd'].squeeze().tolist()[1: real_length - 1])
        # size_list = self.sizevocab.from_seq(self.state['flow_size'].squeeze().tolist()[1: real_length - 1])
        # print('time_before: ', ipd_list)
        # print('size_before: ', size_list)
        self.tran_state(action)

        # real_length = self.state['real_length'].item()
        # ipd_list = self.timevocab.from_seq(self.state['flow_ipd'].squeeze().tolist()[1: real_length - 1])
        # size_list = self.sizevocab.from_seq(self.state['flow_size'].squeeze().tolist()[1: real_length - 1])
        # print('time: ', ipd_list)
        # print('size: ', size_list)

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
        # real_length = self.state['real_length'].item() 
        # ipd_list = self.timevocab.from_seq(self.state['flow_ipd'].squeeze().tolist()[1: real_length - 1])
        # size_list = self.sizevocab.from_seq(self.state['flow_size'].squeeze().tolist()[1: real_length - 1])

        ipd_list = self.real_feat['ipd']
        size_list = self.real_feat['size']
        ipd_tensor = torch.tensor(ipd_list).to(self.device)
        size_tensor = torch.tensor(size_list).to(self.device)

        res = self.eval.evaluate([ipd_tensor, size_tensor])
        print(res)
        
        return res

    def step(self, action):
        self.tran_state(action)
        # self.tran_state(action, ipd_window=[40, 55])

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
        # self.acc = 0

    def reset(self):
        super().reset()
        # print('ori_ipd:', self.real_feat['ipd'])
        # print('ori_size:', self.real_feat['size'])
        self.get_res()
        self.latest = self.constraint_penalty()
        return self.state
    
    def get_res(self,):
        # real_length = self.state['real_length'].item() 
        # ipd_list = self.timevocab.from_seq(self.state['flow_ipd'].squeeze().tolist()[1: real_length - 1])
        # size_list = self.sizevocab.from_seq(self.state['flow_size'].squeeze().tolist()[1: real_length - 1])
        
        # dpdk logic
        pairs = [(b, int(a * 1e6)) for a, b in zip(self.real_feat['ipd'], self.real_feat['size'])]
        send_events(self.sock, pairs)
        ack = receive_events(self.sock)
        if ack is None:
            print('error')
            return -1000
        
        result = self.eval.evaluate([self.real_feat['ipd'], self.real_feat['size'], self.src, self.dst, self.srcport, self.dstport])


        # result = self.eval.evaluate([ipd_list, size_list, self.src, self.dst, self.srcport, self.dstport])
        # print(result)
        return result

    def step(self, action):
        # print('action: ', action)
        # real_length = self.state['real_length'].item()
        # ipd_list = self.timevocab.from_seq(self.state['flow_ipd'].squeeze().tolist()[1: real_length - 1])
        # size_list = self.sizevocab.from_seq(self.state['flow_size'].squeeze().tolist()[1: real_length - 1])
        # print('time_before: ', self.real_feat['ipd'])
        # print('size_before: ', self.real_feat['size'])
        # print('ori_ipd:', self.real_feat['ipd'])
        # last_state = deepcopy(self.state)

        self.tran_state(action)
        
        # print('action: ', action)
        # print('ipd:', self.real_feat['ipd'])
        # print('size:', self.real_feat['size'])
        # print('src:', self.src)
        # print('src_index:', self.src_index)
        # print('original_index:', self.original_index)
        
        # real_length = self.state['real_length'].item()
        # ipd_list = self.timevocab.from_seq(self.state['flow_ipd'].squeeze().tolist()[1: real_length - 1])
        # size_list = self.sizevocab.from_seq(self.state['flow_size'].squeeze().tolist()[1: real_length - 1])
        # print('ipd: ', ipd_list)
        # print('size_nn: ', size_list)
        # print('time: ', self.real_feat['ipd'])
        # print('size:', self.real_feat['size'])
        # self.constraint_penalty()

        result = self.get_res()
        done = False
        reward = 0
        self.step_num += 1

        if self.step_num >= self.rl_args.trainer.max_stop_step + 1:
            done = True
            reward = 0
        
        # reward = -0.01 * self.step_num
        # if self.step_num > 10 and self.state == old_state:
        #     return self.state, -1, True
        
        if result == 0:
            done = True
            reward = 1
        
        latest = self.constraint_penalty()
        dos_penalty = self.latest - latest
        self.latest = latest
        
        step_penalty = 0 if reward == 1 else -0.01

        return self.state, reward + step_penalty + 0. * dos_penalty, done, 1 - result

    def constraint_penalty(self, ):
        # print(self.original_index)
        # ipd_list = self.timevocab.from_seq(self.state['flow_ipd'].squeeze().tolist())
        # total_time = sum(ipd_list[self.original_index[0] + 1: self.original_index[-1]])
        
        total_time = sum(self.real_feat['ipd'][self.original_index[0]: self.original_index[-1] - 1])
        # print(total_time)
        return total_time
    

class TestEnv(BaseEnv):
    def __init__(self, rl_args, bert_args):
        super().__init__(rl_args, bert_args)

    def reset(self):
        super().reset()
        return self.state
    
    def step(self, action):
        # print(self.state['flow_ipd'].shape)
        # print('ori: ', self.state['real_length'])
        real_length = self.state['real_length'].item()
        if action <= 1 or action >= 2 * real_length - 1:
            return self.state, -5, False    # state, reward, done. 
        
        else:
            # print(self.state['flow_ipd'][0, :self.state['real_length']])
            #  (action)
            
            self.tran_state(action)
        
            reward = 1
            done = False
            self.obs_state()
            return self.state, reward, done
        
    def obs_state(self, ):
        print('flow_ipd: ', self.state['flow_ipd'][0, :self.state['real_length'].data])
        print('flow_size: ', self.state['flow_size'][0, :self.state['real_length'].data])


if __name__ == '__main__':
    bert_args_pth = '/home/lzx/NetMasquerade/Pretrain/config/bert.yaml'
    rl_args_pth = '/home/lzx/NetMasquerade/Finetune/config/sac.yaml'
    rl_args = recursive_namespace(read_yaml(rl_args_pth))
    bert_args = recursive_namespace(read_yaml(bert_args_pth))
    env = BaseEnv(rl_args, bert_args)

    env.reset()
    env.step(1)
 
