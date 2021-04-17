import numpy as np
import torch


class ReplayBuffer(object):

    def __init__(self, obs_shape, action_shape, reward_shape, capacity, device):
        self.capacity = capacity
        self.device = device

        self.obses = np.empty((capacity, *obs_shape), dtype=np.float32)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=np.float32)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, *reward_shape), dtype=np.float32)
        self.dones = np.empty((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.last_save = 0
        self.full = False

    def __len__(self):
        return self.capacity if self.full else self.idx

    def add(self, obs, action, reward, next_obs, done):

        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.dones[self.idx], done)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self, batch_size, nth=None):
        idxs = np.random.randint(0, self.capacity if self.full else self.idx, size=batch_size)

        if nth:
            obses = torch.FloatTensor(self.obses[idxs][:, nth]).to(self.device)
            actions = torch.FloatTensor(self.actions[idxs][:, nth]).to(self.device)
            rewards = torch.FloatTensor(self.rewards[idxs][:, nth]).to(self.device)
            next_obses = torch.FloatTensor(self.next_obses[idxs][:, nth]).to(self.device)
            dones = torch.FloatTensor(self.dones[idxs]).to(self.device)
        else:
            obses = torch.FloatTensor(self.obses[idxs]).to(self.device)
            actions = torch.FloatTensor(self.actions[idxs]).to(self.device)
            rewards = torch.FloatTensor(self.rewards[idxs]).to(self.device)
            next_obses = torch.FloatTensor(self.next_obses[idxs]).to(self.device)
            dones = torch.FloatTensor(self.dones[idxs]).to(self.device)

        return obses, actions, rewards, next_obses, dones