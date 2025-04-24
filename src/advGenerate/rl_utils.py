from tqdm import tqdm
import numpy as np
import torch
import collections
import random
import matplotlib.pyplot as plt
import socket
import struct


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity) 

    def add(self, state, action, reward, next_state, done): 
        self.buffer.append((state, action, reward, next_state, done)) 

    def sample(self, batch_size): 
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        if isinstance(state[0], dict):
            keys = state[0].keys()
            # for key, value in state[0].items():
            #     print(key, value.shape)
                
            state = {key: torch.cat([s[key].unsqueeze(0) for s in state], dim=0) for key in keys}

        if isinstance(next_state[0], dict):
            keys = next_state[0].keys()
            next_state = {key: torch.cat([s[key].unsqueeze(0) for s in next_state]) for key in keys}
        
        return state, action, reward, next_state, done 

    def size(self): 
        return len(self.buffer)

def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0)) 
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(a[:window_size-1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))

def train_on_policy_agent(env, agent, num_episodes):
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
                state = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done, _ = env.step(action)
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)
                    state = next_state
                    episode_return += reward
                return_list.append(episode_return)
                agent.update(transition_dict)
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list

def train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size):
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                state = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done, _ = env.step(action)
                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, 'dones': b_d}
                        agent.update(transition_dict)
                return_list.append(episode_return)
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list


def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)


# def generate_mask(real_length, pad_length, mask_index):
#     seq_idx = torch.arange(pad_length, device=real_length.device).unsqueeze(0)
#     if len(real_length.shape) == 1 and real_length.shape[0] > 1:
#         seq_idx = seq_idx.repeat(real_length.shape[0], 1)
#         real_length_expanded = real_length.expand_as(seq_idx)
#     else:
#         real_length_expanded = real_length.expand(seq_idx.shape)

#     expand_mask_index = torch.zeros(mask_index.shape[0], mask_index.shape[1] * 2 + 1, dtype=mask_index.dtype, device=mask_index.device)
#     col = torch.arange(mask_index.shape[1], device=mask_index.device) * 2 + 1
#     expand_mask_index[:, col] = mask_index
#     expand_mask_index[:, torch.arange(mask_index.shape[1] * 2 + 1) % 2 == 0] = 1
#     mask = (seq_idx > 3) & (seq_idx < real_length_expanded - 2) & expand_mask_index

#     return mask

def generate_mask(real_length, pad_length, mask_index):
    if real_length.dim() == 0:
        real_length = real_length.unsqueeze(0)

    seq_idx = torch.arange(pad_length, device=real_length.device).unsqueeze(0)

    if len(real_length.shape) == 1 and real_length.shape[0] > 1:
        seq_idx = seq_idx.repeat(real_length.shape[0], 1)
        real_length_expanded = real_length.unsqueeze(1).expand_as(seq_idx)
    else:
        real_length_expanded = real_length.expand(seq_idx.shape)
    
    if mask_index.dim() == 1:
        mask_index = mask_index.unsqueeze(0)
    
    expand_mask_index = torch.zeros(mask_index.shape[0], mask_index.shape[1] * 2 + 1, dtype=mask_index.dtype, device=mask_index.device)
    col = torch.arange(mask_index.shape[1], device=mask_index.device) * 2 + 1
    expand_mask_index[:, col] = mask_index
    expand_mask_index[:, torch.arange(mask_index.shape[1] * 2 + 1) % 2 == 0] = 1
    
    mask = (seq_idx > 3) & (seq_idx < real_length_expanded - 2) & expand_mask_index

    return mask.squeeze()

def plot_train_rewards(rewards, save_path, ylabel):
    # Set modern aesthetics
    plt.style.use('seaborn-darkgrid')
    
    # Create a color palette
    palette = plt.get_cmap('Set2')
    
    # Create the plot
    plt.figure(figsize=(10,5))
    plt.plot(rewards, color=palette(0))
    plt.title('Training Reward Over Episodes')
    plt.xlabel('Episodes')
    plt.ylabel(ylabel)
    
    # Save the figure
    plt.savefig(save_path)
    plt.close()

def socket_conn(HOST, PORT):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((HOST, PORT))
    return s

def send_events(sock, events):
    """
    events: List[Tuple[int,int]]  # (size, timestamp)
    """
    N = len(events)
    header = struct.pack('!I', N)
    body   = b''.join(struct.pack('!IQ', sz, ts)
                      for sz, ts in events)
    sock.sendall(header + body)  

def receive_events(sock):
    ack_buf = b''
    while len(ack_buf) < 4:
        chunk = sock.recv(4 - len(ack_buf))
        if not chunk:
            raise ConnectionError("controller: connection closed unexpectedly")
        ack_buf += chunk
    (ack,) = struct.unpack('!I', ack_buf)
    return ack


if __name__ == '__main__':

    a = [1, 3, 4]
    b = torch.zeros(1, 5)
    b[0, a] = 1

    real_length = torch.tensor(3 * 2 + 1, device='cpu')
    pad_length = torch.tensor(5 * 2 + 1, device='cpu')
    mask = generate_mask(real_length, pad_length, b)

    print(mask)
