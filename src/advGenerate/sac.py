"""Discrete Soft Actor-Critic agent used by NetMasquerade."""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from advGenerate.rl_utils import generate_mask


class _SequenceHead(nn.Module):
    def __init__(self, timevocab_size, sizevocab_size, embedding_dim,
                 hidden_dim, state_dim):
        super().__init__()
        self.state_dim = int(state_dim)
        self.ipd_embedding = nn.Embedding(timevocab_size, embedding_dim)
        self.size_embedding = nn.Embedding(sizevocab_size, embedding_dim)
        self.rnn = nn.GRU(embedding_dim * 2, hidden_dim, batch_first=True)
        self.reduce = nn.Linear(hidden_dim, 1)
        self.actions = nn.Linear(self.state_dim, 2 * self.state_dim + 1)

    def scores(self, state):
        ipd = self.ipd_embedding(state["flow_ipd"])
        size = self.size_embedding(state["flow_size"])
        encoded, _ = self.rnn(torch.cat((ipd, size), dim=-1))
        scores = self.actions(self.reduce(encoded).squeeze(-1))
        legal = generate_mask(state["real_length"], self.state_dim,
                              state["src_index"])
        if legal.dim() == 1:
            legal = legal.unsqueeze(0)
        return scores.masked_fill(~legal, -1e9)


class PolicyNet(_SequenceHead):
    def forward(self, state):
        return F.softmax(self.scores(state), dim=-1)


class QValueNet(_SequenceHead):
    def forward(self, state):
        return self.scores(state)


class SAC:
    def __init__(self, timevocab_size, sizevocab_size, embedding_dim,
                 state_dim, hidden_dim, actor_lr, critic_lr, alpha_lr,
                 target_entropy, tau, gamma, device):
        parameters = (timevocab_size, sizevocab_size, embedding_dim,
                      hidden_dim, state_dim)
        self.actor = PolicyNet(*parameters).to(device)
        self.critic_1 = QValueNet(*parameters).to(device)
        self.critic_2 = QValueNet(*parameters).to(device)
        self.target_critic_1 = QValueNet(*parameters).to(device)
        self.target_critic_2 = QValueNet(*parameters).to(device)
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=critic_lr)
        self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float32,
                                      device=device, requires_grad=True)
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)
        self.target_entropy = float(target_entropy)
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.device = device

    def transform_state(self, state):
        transformed = {}
        for name, value in state.items():
            value = value.to(self.device)
            if name != "real_length" and value.dim() == 1:
                value = value.unsqueeze(0)
            elif name == "real_length" and value.dim() == 0:
                value = value.unsqueeze(0)
            transformed[name] = value
        return transformed

    def take_action(self, state):
        probabilities = self.actor(self.transform_state(state))
        return int(torch.distributions.Categorical(probabilities).sample().item())

    def take_deterministic_action(self, state):
        with torch.no_grad():
            probabilities = self.actor(self.transform_state(state))
        return int(probabilities.argmax(dim=-1).item())

    def action_value(self, state, action):
        with torch.no_grad():
            state = self.transform_state(state)
            conservative = torch.minimum(self.critic_1(state), self.critic_2(state))
        return float(conservative[0, int(action)].item())

    def calc_target(self, rewards, next_states, dones):
        with torch.no_grad():
            probabilities = self.actor(next_states)
            log_probabilities = torch.log(probabilities + 1e-8)
            entropy = -(probabilities * log_probabilities).sum(dim=1, keepdim=True)
            q_value = torch.minimum(self.target_critic_1(next_states),
                                    self.target_critic_2(next_states))
            expected_q = (probabilities * q_value).sum(dim=1, keepdim=True)
            next_value = expected_q + self.log_alpha.exp() * entropy
            return rewards + self.gamma * next_value * (1 - dones)

    def _soft_update(self, network, target):
        for target_parameter, parameter in zip(target.parameters(),
                                                network.parameters()):
            target_parameter.data.mul_(1.0 - self.tau)
            target_parameter.data.add_(parameter.data, alpha=self.tau)

    def update(self, transition):
        states = self.transform_state(transition["states"])
        next_states = self.transform_state(transition["next_states"])
        actions = torch.as_tensor(transition["actions"], dtype=torch.long,
                                  device=self.device).view(-1, 1)
        rewards = torch.as_tensor(transition["rewards"], dtype=torch.float32,
                                  device=self.device).view(-1, 1)
        dones = torch.as_tensor(transition["dones"], dtype=torch.float32,
                                device=self.device).view(-1, 1)
        target = self.calc_target(rewards, next_states, dones)

        for critic, optimizer in ((self.critic_1, self.critic_1_optimizer),
                                  (self.critic_2, self.critic_2_optimizer)):
            loss = F.mse_loss(critic(states).gather(1, actions), target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        probabilities = self.actor(states)
        log_probabilities = torch.log(probabilities + 1e-8)
        entropy = -(probabilities * log_probabilities).sum(dim=1, keepdim=True)
        q_value = torch.minimum(self.critic_1(states),
                                self.critic_2(states)).detach()
        expected_q = (probabilities * q_value).sum(dim=1, keepdim=True)
        actor_loss = (-self.log_alpha.exp().detach() * entropy - expected_q).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = (self.log_alpha.exp() *
                      (entropy.detach() - self.target_entropy)).mean()
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()
        self._soft_update(self.critic_1, self.target_critic_1)
        self._soft_update(self.critic_2, self.target_critic_2)

    def save_model(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.actor.state_dict(), path / "actor.pth")
        torch.save(self.critic_1.state_dict(), path / "critic_1.pth")
        torch.save(self.critic_2.state_dict(), path / "critic_2.pth")

    def load_model(self, path):
        path = Path(path)
        self.actor.load_state_dict(torch.load(path / "actor.pth",
                                              map_location=self.device))
        self.critic_1.load_state_dict(torch.load(path / "critic_1.pth",
                                                 map_location=self.device))
        self.critic_2.load_state_dict(torch.load(path / "critic_2.pth",
                                                 map_location=self.device))
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())

    def eval_stop(self, state, threshold, action=None):
        if action is None:
            action = self.take_deterministic_action(state)
        return self.action_value(state, action) > float(threshold)
