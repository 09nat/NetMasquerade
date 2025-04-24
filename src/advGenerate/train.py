import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.nn as nn
import argparse
import random
import gym
import time
import numpy as np
from environments.environment import *
from Pretrain.utils import *
from Pretrain.dataset import TimeVocab, SizeVocab
from rl_utils import *
from sac import SAC
from environments.environment import *
from PacketProcessing import Flow
import dill as pickle

torch.set_printoptions(threshold=10000)


class Trainer:
    def __init__(self, rl_args, bert_args):
        self.rl_args = rl_args
        self.bert_args = bert_args
        self.device = set_device(self.rl_args.trainer.device)

        self.env = None

        # random.seed(512)
        # np.random.seed(512)
        # torch.manual_seed(512)

        self.replay_buffer = ReplayBuffer(self.rl_args.trainer.buffer_size)
        self.agent = SAC(self.rl_args.model.timevocab_size, self.rl_args.model.sizevocab_size,  
                    self.rl_args.model.embedding_dim, self.rl_args.model.state_dim, self.rl_args.model.hidden_dim, 
                    self.rl_args.model.actor_lr, self.rl_args.model.critic_lr, 
                    self.rl_args.model.alpha_lr, self.rl_args.model.target_entropy, self.rl_args.model.tau, 
                    self.rl_args.model.gamma, self.device)

        
    def train_off_policy_agent(self, ): 
        if self.rl_args.trainer.env_name == 'KitSune':
            self.env = KitSuneEnv(rl_args, bert_args)
        elif self.rl_args.trainer.env_name == 'LSTM':
            self.env = LSTMEnv(rl_args, bert_args)
        elif self.rl_args.trainer.env_name == 'NetBeacon':
            self.env = NetBeaconEnv(rl_args, bert_args)
        elif self.rl_args.trainer.env_name == 'SVM':
            self.env = SVMEnv(rl_args, bert_args)
        elif self.rl_args.trainer.env_name == 'Whisper':
            self.env = WhisperEnv(rl_args, bert_args)
        elif self.rl_args.trainer.env_name == 'MLP':
            self.env = MLPEnv(rl_args, bert_args)
            
            
        num_episodes = self.rl_args.trainer.episodes
        return_list = []
        asr_list = []
        qvalues = []
        best_model_asr = -1
        state_record = []
        with tqdm(total=num_episodes, desc="Training Progress") as pbar:
            for i_episode in range(num_episodes):
                episode_return = 0
                state = self.env.reset()
                done = False
                state_record.append(state)
                while not done:
                    action = self.agent.take_action(state)
                    next_state, reward, done, result = self.env.step(action)

                    if done:
                        with torch.no_grad():
                            q_values_1 = self.agent.critic_1(state)
                            q_values_2 = self.agent.critic_2(state)
                        max_q_value = torch.max(q_values_1.squeeze()[action], q_values_2.squeeze()[action])
                        # max_q_value = torch.max(torch.min(q_values_1, q_values_2))
                        qvalues.append(max_q_value.item())

                    self.replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward
                    if self.replay_buffer.size() > self.rl_args.trainer.minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = self.replay_buffer.sample(self.rl_args.trainer.batch_size)
                        transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, 'dones': b_d}
                        self.agent.update(transition_dict)
                return_list.append(episode_return)
                asr_list.append(result)
                if (i_episode + 1) % 5 == 0:
                    pbar.set_postfix({
                                        'episode': i_episode + 1,
                                        'average_return': f"{np.mean(return_list[-100:]):.4f}",
                                        'average_asr': f"{np.mean(asr_list[-100:]):.4f}"
                                    })

                pbar.update(1)
                
                if (i_episode + 1) % self.rl_args.trainer.test_episodes == 0:
                    asr = np.mean(asr_list[-self.rl_args.trainer.test_episodes:])
                    for item in state_record[-self.rl_args.trainer.test_episodes:-self.rl_args.trainer.test_episodes + 5]:
                        print(item['flow_ipd'][:5])
                    print('asr:', asr)
                    if asr > best_model_asr:
                        best_model_asr = asr
                        # self.agent.save_model(self.rl_args.trainer.model_save_pth)
                        print('best model saved')
                    # print(trainer.agent.actor.fc1.weight.data)
                    # print(asr_list[-self.rl_args.trainer.test_episodes:])

                if (i_episode + 1) % 100 == 0:
                    ...
                    if self.rl_args.trainer.fig_save_pth is not None:
                        plot_train_rewards(qvalues, save_path=self.rl_args.trainer.fig_save_pth.format(i_episode + 1), ylabel='Critic Max Qvalue')
                    if best_model_asr < 0:
                        # self.agent.save_model(self.rl_args.trainer.model_save_pth)
                        print('no model before, model on {} saved'.format(i_episode + 1))
                        print('asr:', np.mean(asr_list[-100:]))
                    # name = self.rl_args.trainer.model_save_pth.split('/')[-2]
                    # with open('/home/lzx/NetMasquerade/reward_result/' + name + '_qvalues3.pkl', 'wb') as f:
                    #     pickle.dump(qvalues, f)
        return return_list
    
    
    def evaluate(self, eval_episodes=10):
        if self.rl_args.trainer.env_name == 'KitSune':
            self.env = KitSuneEnv(rl_args, bert_args, 'test')
        elif self.rl_args.trainer.env_name == 'LSTM':
            self.env = LSTMEnv(rl_args, bert_args, 'test')
        elif self.rl_args.trainer.env_name == 'NetBeacon':
            self.env = NetBeaconEnv(rl_args, bert_args, 'test')
        elif self.rl_args.trainer.env_name == 'SVM':
            self.env = SVMEnv(rl_args, bert_args, 'test')
        elif self.rl_args.trainer.env_name == 'Whisper':
            self.env = WhisperEnv(rl_args, bert_args, 'test')
        elif self.rl_args.trainer.env_name == 'MLP':
            self.env = MLPEnv(rl_args, bert_args, 'test')
            
        total_result= []
        q_values = []
        step_list = []
        
        if self.rl_args.trainer.flow_save_pth is not None:
            packet_feature = {
                'ipd': [],
                'size': [],
                'ori': [],
                'src': [],
                'first_timestamp': []
            }

        with tqdm(range(eval_episodes), desc='evaluating...') as pbar:
            for _ in pbar:
                state = self.env.reset()
                done = False
                # print(state['flow_ipd'][0, :5])
                while not done:
                    action = self.agent.take_deterministic_action(state)
                    # action = self.agent.take_action(state)
                    next_state, reward, done, result = self.env.step(action)
                    if done:
                        q_values_1 = self.agent.critic_1(state)
                        q_values_2 = self.agent.critic_2(state)
                        max_q_value = torch.max(q_values_1.squeeze()[action], q_values_2.squeeze()[action])
                        step_list.append(self.env.step_num)
                        q_values.append(max_q_value)
                    state = next_state

                if self.rl_args.trainer.flow_save_pth is not None:
                    packet_feature['ipd'].append(self.env.real_feat['ipd'])
                    packet_feature['size'].append(self.env.real_feat['size'])
                    packet_feature['ori'].append(self.env.original_index)
                    packet_feature['src'].append(self.env.src_index)
                    packet_feature['first_timestamp'].append(self.env.flow.timestp[0])

                total_result.append(result)
                average_result = np.mean(total_result)
                pbar.set_postfix(average_result="{:.4f}".format(average_result))
        print('mean q value: ', torch.mean(torch.stack(q_values)))
        print('average step: ', np.mean(step_list))
        
        if self.rl_args.trainer.flow_save_pth is not None:
            with open(self.rl_args.trainer.flow_save_pth, 'wb') as f:
                pickle.dump(packet_feature, f)
                 
    
        # with open('/home/lzx/NetMasquerade/Analysis/adv_syn.pkl', 'wb') as f:
            # pickle.dump(self.env.adv_flow, f)   
            # print('adversarial flow saved')
    
        return total_result

    def evaluate_without_feedback(self, threshold, stop_step, eval_episodes=10):
        if self.rl_args.trainer.env_name == 'KitSune':
            self.env = KitSuneEnv(rl_args, bert_args, 'test')
        elif self.rl_args.trainer.env_name == 'LSTM':
            self.env = LSTMEnv(rl_args, bert_args, 'test')
        elif self.rl_args.trainer.env_name == 'NetBeacon':
            self.env = NetBeaconEnv(rl_args, bert_args, 'test')
        elif self.rl_args.trainer.env_name == 'SVM':
            self.env = SVMEnv(rl_args, bert_args, 'test')
        elif self.rl_args.trainer.env_name == 'Whisper':
            self.env = WhisperEnv(rl_args, bert_args, 'test')
        elif self.rl_args.trainer.env_name == 'MLP':
            self.env = MLPEnv(rl_args, bert_args, 'test')
            
        total_result = []
        
        if self.rl_args.trainer.flow_save_pth is not None:
            packet_feature = {
                'ipd': [],
                'size': [],
                'ori': [],
                'src': [],
                'first_timestamp': []
            }
        
        with tqdm(range(eval_episodes), desc='evaluating...') as pbar:
            for _ in pbar:
                state = self.env.reset()
                step = 0
                while 1:
                    step += 1
                    # real_length = state['real_length'].item()
                    # action = random.randint(3, 2 * real_length - 2)
                    action = self.agent.take_deterministic_action(state)
                    next_state, reward, done, result = self.env.step(action)
                    if step >= stop_step + 1 or self.agent.eval_stop(state, threshold, action=action):
                        total_result.append(result)
                        # if self.rl_args.trainer.flow_save_pth is not None:
                        #     packet_feature['ipd'].append(self.env.real_feat['ipd'])
                        #     packet_feature['size'].append(self.env.real_feat['size'])
                        #     packet_feature['ori'].append(self.env.original_index)
                        #     packet_feature['src'].append(self.env.src_index)
                        #     packet_feature['first_timestamp'].append(self.env.flow.timestp[0])
                        break
                    state = next_state
                # print(step)
                average_reward = np.mean(total_result)
                pbar.set_postfix(average_reward="{:.4f}".format(average_reward))
        
        if self.rl_args.trainer.flow_save_pth is not None:
            with open(self.rl_args.trainer.flow_save_pth, 'wb') as f:
                pickle.dump(packet_feature, f)
        return total_result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rl_pth', type=str, default='/home/lzx/NetMasquerade/Finetune/config/sac.yaml',
                        help='path to load config', )
    parser.add_argument('--bert_pth', type=str, default='/home/lzx/NetMasquerade/Pretrain/config/bert.yaml',
                        help='path to load config', )
    args = parser.parse_args()
    rl_args = recursive_namespace(read_yaml(args.rl_pth))
    bert_args = recursive_namespace(read_yaml(args.bert_pth))

    trainer = Trainer(rl_args, bert_args)

    # train
    return_list = trainer.train_off_policy_agent()

    # test
    # trainer.agent.load_model(trainer.rl_args.trainer.model_save_pth)
    # start_time = time.perf_counter()
    # result_list = trainer.evaluate(eval_episodes=rl_args.trainer.test_episodes)
    # result_list = trainer.evaluate_without_feedback(trainer.rl_args.trainer.stop_threshold, trainer.rl_args.trainer.max_stop_step, 
                                                    #    eval_episodes=rl_args.trainer.test_episodes)

    # end_time = time.perf_counter()
    # elapsed_time = end_time - start_time
    # print('elapsed time: ', elapsed_time)
    # print('average_reward: ', np.sum(result_list) / len(result_list))