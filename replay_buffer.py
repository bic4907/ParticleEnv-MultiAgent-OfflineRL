import numpy as np
import torch
import json, os

from utils.train import make_dir


class ReplayBuffer(object):

    def __init__(self, obs_shape, action_shape, reward_shape, dones_shape, capacity, device):
        self.capacity = capacity
        self.device = device

        self.obses = np.empty((capacity, *obs_shape), dtype=np.float32)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=np.float32)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, *reward_shape), dtype=np.float32)
        self.dones = np.empty((capacity, *dones_shape), dtype=np.float32)

        self.idx = 0
        self.full = False

    def __len__(self):
        return self.capacity if self.full else self.idx

    def add(self, obs, action, reward, next_obs, dones):

        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.dones[self.idx], dones)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self, batch_size, nth=None):
        idxs = np.random.randint(0, self.capacity if self.full else self.idx, size=batch_size)

        if nth:
            obses = torch.FloatTensor(self.obses[idxs][:, nth]).to(self.device)
            actions = torch.FloatTensor(self.actions[idxs][:, nth]).to(self.device)
            rewards = torch.FloatTensor(self.rewards[idxs][:, nth]).to(self.device)
            next_obses = torch.FloatTensor(self.next_obses[idxs][:, nth]).to(self.device)
            dones = torch.FloatTensor(self.dones[idxs][:, nth]).to(self.device)
        else:
            obses = torch.FloatTensor(self.obses[idxs]).to(self.device)
            actions = torch.FloatTensor(self.actions[idxs]).to(self.device)
            rewards = torch.FloatTensor(self.rewards[idxs]).to(self.device)
            next_obses = torch.FloatTensor(self.next_obses[idxs]).to(self.device)
            dones = torch.FloatTensor(self.dones[idxs]).to(self.device)

        return obses, actions, rewards, next_obses, dones

    def save(self, root_dir, step) -> None:
        make_dir(root_dir, 'buffer') if root_dir else None
        length = self.capacity if self.full else self.idx
        path = os.path.join(root_dir, 'buffer')

        np.save(os.path.join(path, 'state.npy'), self.obses)
        np.save(os.path.join(path, 'next_state.npy'), self.next_obses)
        np.save(os.path.join(path, 'action.npy'), self.actions)
        np.save(os.path.join(path, 'reward.npy'), self.rewards)
        np.save(os.path.join(path, 'done.npy'), self.dones)

        info = dict()
        info['idx'] = self.idx
        info['capacity'] = self.capacity
        info['step'] = step
        info['size'] = length
        json.dumps('info.txt', indent=4, sort_keys=True)

