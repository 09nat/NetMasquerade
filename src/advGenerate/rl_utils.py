"""Replay storage and legal-action masking for discrete SAC."""

import collections
import random

import torch


def clone_state(state, device="cpu"):
    """Detach a state snapshot so later environment edits cannot overwrite it."""
    return {name: value.detach().clone().to(device)
            for name, value in state.items()}


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=int(capacity))

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((clone_state(state), int(action), float(reward),
                            clone_state(next_state), bool(done)))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, int(batch_size))
        states, actions, rewards, next_states, dones = zip(*transitions)
        state_batch = {key: torch.stack([state[key] for state in states])
                       for key in states[0]}
        next_batch = {key: torch.stack([state[key] for state in next_states])
                      for key in next_states[0]}
        return state_batch, actions, rewards, next_batch, dones

    def __len__(self):
        return len(self.buffer)


def generate_mask(real_length, state_dim, src_index):
    """Return a boolean mask for ``2 * state_dim + 1`` actions.

    ``2 * position`` inserts before a payload/end token.  ``2 * position + 1``
    modifies an attacker-owned payload IPD.  Start/end/padding tokens cannot be
    modified, and insertion is disabled when the token sequence is full.
    """
    if real_length.dim() == 0:
        real_length = real_length.unsqueeze(0)
    if src_index.dim() == 1:
        src_index = src_index.unsqueeze(0)
    batch = real_length.shape[0]
    mask = torch.zeros((batch, 2 * state_dim + 1), dtype=torch.bool,
                       device=real_length.device)
    positions = torch.arange(state_dim, device=real_length.device).unsqueeze(0)
    lengths = real_length.view(-1, 1)

    can_insert = ((positions >= 1) & (positions <= lengths - 1) &
                  (lengths < state_dim))
    can_modify = ((positions >= 1) & (positions < lengths - 1) &
                  src_index.bool())
    even_columns = 2 * torch.arange(state_dim, device=real_length.device)
    odd_columns = even_columns + 1
    mask[:, even_columns] = can_insert
    mask[:, odd_columns] = can_modify
    if not torch.all(mask.any(dim=1)):
        raise ValueError("observation has no legal packet-edit action")
    return mask.squeeze(0) if batch == 1 else mask
