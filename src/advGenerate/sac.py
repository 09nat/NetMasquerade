import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from rl_utils import *
# consider bi-gru?
# gru + mask (with higher time complexity!)

torch.set_printoptions(threshold=10000)
class PolicyNet(torch.nn.Module):
    def __init__(self, timevocab_size, sizevocab_size, embedding_dim, hidden_dim, state_dim):
        super(PolicyNet, self).__init__()
        self.state_dim = state_dim
        self.ipd_embedding = nn.Embedding(timevocab_size, embedding_dim)
        self.size_embedding = nn.Embedding(sizevocab_size, embedding_dim)
        self.rnn = nn.GRU(embedding_dim * 2, hidden_dim, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, 1)
        self.fc2 = nn.Linear(state_dim, 2 * state_dim + 1)  # 1025 here!

    def forward(self, state):
        ipd_embed = self.ipd_embedding(state['flow_ipd'])
        size_embed = self.size_embedding(state['flow_size'])
        mask = generate_mask(state['real_length'] * 2 + 1, self.state_dim * 2 + 1, state['src_index'])
        # print('mask:', mask)
        # print(mask.shape)
        combined_seq = torch.cat((ipd_embed, size_embed), dim=-1)   # [bs, seq_len, embed_dim * 2]
        out, _ = self.rnn(combined_seq)
        scores = self.fc1(out).squeeze(-1)
        scores = self.fc2(scores)
        # print(scores.shape)
        scores.masked_fill_(mask == 0, -1e9)
        return F.softmax(scores, dim=-1)

# another version: mlp + mask
# class PolicyNet(torch.nn.Module):
#     def __init__(self, state_dim, hidden_dim, action_dim):
#         super(PolicyNet, self).__init__()
#         self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
#         self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

#     def forward(self, x, mask):
#         # print('x:', x.dtype)
#         # print(self.fc1.dtype)
#         x = F.relu(self.fc1(x))
#         scores = self.fc2(x)
#         scores.masked_fill_(mask == 0, -float('inf'))

#         return F.softmax(scores, dim=-1)


class QValueNet(torch.nn.Module):
    def __init__(self, timevocab_size, sizevocab_size, embedding_dim, hidden_dim, state_dim):
        super(QValueNet, self).__init__()
        self.state_dim = state_dim
        self.ipd_embedding = nn.Embedding(timevocab_size, embedding_dim)
        self.size_embedding = nn.Embedding(sizevocab_size, embedding_dim)
        self.rnn = nn.GRU(embedding_dim * 2, hidden_dim, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, 1)
        self.fc2 = nn.Linear(state_dim, 2 * state_dim + 1)

    def forward(self, state):
        ipd_embed = self.ipd_embedding(state['flow_ipd'])
        size_embed = self.size_embedding(state['flow_size'])
        mask = generate_mask(state['real_length'] * 2 + 1, self.state_dim * 2 + 1, state['src_index'])
        combined_seq = torch.cat((ipd_embed, size_embed), dim=-1)   # [bs, seq_len, embed_dim * 2]
        out, _ = self.rnn(combined_seq)
        scores = self.fc1(out).squeeze(-1)
        scores = self.fc2(scores)
        # print(scores.shape)
        scores.masked_fill_(mask == 0, -1e9)
        return scores

# class QValueNet(torch.nn.Module):
#     def __init__(self, timevocab_size, sizevocab_size, embedding_dim, hidden_dim, seq_len):
#         super(QValueNet, self).__init__()
#         self.rnn = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
#         self.fc1 = nn.Linear(hidden_dim, 1)
#         self.fc2 = nn.Linear(seq_len, 2 * seq_len + 1)  # 1025 here!

#     def forward(self, state, mask):
#         out, _ = self.rnn(x)
#         scores = self.fc1(out).squeeze(-1)
#         scores = self.fc2(scores)
#         scores.masked_fill_(mask == 0, -float('inf'))
#         return scores

# class QValueNet(torch.nn.Module):
#     def __init__(self, state_dim, hidden_dim, action_dim):
#         super(QValueNet, self).__init__()
#         self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
#         self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         return self.fc2(x)


class SAC:
    def __init__(self, timevocab_size, sizevocab_size, embedding_dim, state_dim, hidden_dim, actor_lr, critic_lr,
                 alpha_lr, target_entropy, tau, gamma, device):
        self.actor = PolicyNet(timevocab_size, sizevocab_size, embedding_dim, hidden_dim, state_dim).to(device)
        self.critic_1 = QValueNet(timevocab_size, sizevocab_size, embedding_dim, hidden_dim, state_dim).to(device)
        self.critic_2 = QValueNet(timevocab_size, sizevocab_size, embedding_dim, hidden_dim, state_dim).to(device)
        self.target_critic_1 = QValueNet(timevocab_size, sizevocab_size, embedding_dim, hidden_dim,
                                         state_dim).to(device)
        self.target_critic_2 = QValueNet(timevocab_size, sizevocab_size, embedding_dim, hidden_dim,
                                         state_dim).to(device)

        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(),
                                                   lr=critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(),
                                                   lr=critic_lr)
        self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],
                                                    lr=alpha_lr)
        self.target_entropy = target_entropy
        self.gamma = gamma
        self.tau = tau
        self.device = device

    def transform_state(self, states_dict):
        states_dict['flow_ipd'] = states_dict['flow_ipd'].to(self.device)
        states_dict['flow_size'] = states_dict['flow_size'].to(self.device)
        states_dict['real_length'] = states_dict['real_length'].to(self.device)
        states_dict['src_index'] = states_dict['src_index'].to(self.device)
        return states_dict

    def take_action(self, state):
        state = self.transform_state(state)
        # state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.actor(state)
        # print('probs: ', probs[0][:20])
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()
    
    def take_deterministic_action(self, state):
        state = self.transform_state(state)
        with torch.no_grad():
            probs = self.actor(state)
        # print('probs: ', probs)
        action = torch.argmax(probs, dim=-1)
        return action.item()

    def calc_target(self, rewards, next_states, dones):
        next_probs = self.actor(next_states)
        next_log_probs = torch.log(next_probs + 1e-8)
        entropy = -torch.sum(next_probs * next_log_probs, dim=1, keepdim=True)
        q1_value = self.target_critic_1(next_states)
        q2_value = self.target_critic_2(next_states)
        min_qvalue = torch.sum(next_probs * torch.min(q1_value, q2_value),
                               dim=1,
                               keepdim=True)
        next_value = min_qvalue + self.log_alpha.exp() * entropy
        td_target = rewards + self.gamma * next_value * (1 - dones)
        return td_target

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(),
                                       net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) +
                                    param.data * self.tau)

    def update(self, transition_dict):
        states = self.transform_state(transition_dict['states'])

        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(
            self.device)
        
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        
        next_states = self.transform_state(transition_dict['next_states'])

        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)

        # update q
        td_target = self.calc_target(rewards, next_states, dones)

        critic_1_q_values = self.critic_1(states).gather(1, actions)
        critic_1_loss = torch.mean(
            F.mse_loss(critic_1_q_values, td_target.detach()))
        critic_2_q_values = self.critic_2(states).gather(1, actions)
        critic_2_loss = torch.mean(
            F.mse_loss(critic_2_q_values, td_target.detach()))
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        # udpate actor
        probs = self.actor(states)
        log_probs = torch.log(probs + 1e-8)
        # calculate entropy
        entropy = -torch.sum(probs * log_probs, dim=1, keepdim=True)  #
        q1_value = self.critic_1(states)
        q2_value = self.critic_2(states)

        # print(torch.max(torch.min(q1_value, q2_value)))

        min_qvalue = torch.sum(probs * torch.min(q1_value, q2_value),
                               dim=1,
                               keepdim=True)
        actor_loss = torch.mean(-self.log_alpha.exp() * entropy - min_qvalue)
        # print('actor_loss: ', actor_loss)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # update alpha
        alpha_loss = torch.mean(
            (entropy - self.target_entropy).detach() * self.log_alpha.exp())
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        self.soft_update(self.critic_1, self.target_critic_1)
        self.soft_update(self.critic_2, self.target_critic_2)

    def save_model(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

        torch.save(self.actor.state_dict(), os.path.join(path, 'actor.pth'))
        torch.save(self.critic_1.state_dict(), os.path.join(path, 'critic_1.pth'))
        torch.save(self.critic_2.state_dict(), os.path.join(path, 'critic_2.pth'))

    def load_model(self, path):
        self.actor.load_state_dict(torch.load(os.path.join(path, 'actor.pth'), map_location=self.device))
        self.critic_1.load_state_dict(torch.load(os.path.join(path, 'critic_1.pth'), map_location=self.device))
        self.critic_2.load_state_dict(torch.load(os.path.join(path, 'critic_2.pth'), map_location=self.device))
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        self.actor.eval()
        self.critic_1.eval()
        self.critic_2.eval()

    def eval_stop(self, state, threshold, action=None):
        state = self.transform_state(state)
        q_values_1 = self.critic_1(state)
        q_values_2 = self.critic_2(state)
        # print(torch.min(q_values_1, q_values_2))
        # print(torch.max(torch.min(q_values_1, q_values_2)))
        if action is None:
            max_q_value = torch.max(torch.min(q_values_1, q_values_2))
        else:
            max_q_value = torch.max(q_values_1.squeeze()[action], q_values_2.squeeze()[action])
        # print(q_values_1.argmax().item(), q_values_2.argmax().item(), action)
        # print(q_values_1[:10])
        # print(q_values_2[:10])
        # print(max_q_value.item(), torch.max(torch.min(q_values_1, q_values_2)).item())
        return max_q_value.item() > threshold
