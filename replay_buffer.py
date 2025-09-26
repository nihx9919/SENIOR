import numpy as np
import torch
import utils
from sklearn.neighbors import KernelDensity


class ReplayBuffer(object):
    """Buffer to store environment transitions."""

    def __init__(self, obs_shape, action_shape, capacity, device,
                 time_decrease, pre_buffer_capacity,
                 window=1):
        self.capacity = capacity
        self.device = device
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        # the proprioceptive obs is stored as float32, pixels obs as uint8
        obs_dtype = np.float32 if len(obs_shape) == 1 else np.uint8

        self.obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones_no_max = np.empty((capacity, 1), dtype=np.float32)

        self.window = window

        self.idx = 0
        self.last_save = 0
        self.full = False

        # for preference-guided exploration
        self.time_de = time_decrease
        self.pre_buffer_capacity = pre_buffer_capacity

        self.obses_pre_buffer = np.empty((pre_buffer_capacity, 3), dtype=obs_dtype)
        self.pre_buffer_idx = 0
        self.pre_buffer_full = False
        self.pre_distribution = None
        self.density_distribution = None

        # intrinsic buffer
        self.curiosity_buffer_size = 1000
        self.curiosity_exp_ratio = 0.3
        self.normal_exp_idx = []

        self.obses_int = np.empty((self.curiosity_buffer_size, *obs_shape), dtype=obs_dtype)
        self.next_obses_int = np.empty((self.curiosity_buffer_size, *obs_shape), dtype=obs_dtype)
        self.actions_int = np.empty((self.curiosity_buffer_size, *action_shape), dtype=np.float32)
        self.rewards_int = np.empty((self.curiosity_buffer_size, 1), dtype=np.float32)
        self.not_dones_int = np.empty((self.curiosity_buffer_size, 1), dtype=np.float32)
        self.not_dones_no_max_int = np.empty((self.curiosity_buffer_size, 1), dtype=np.float32)

    def __len__(self):
        return self.capacity if self.full else self.idx

    def add(self, obs, action, reward, next_obs, done, done_no_max):
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.not_dones[self.idx], not done)
        np.copyto(self.not_dones_no_max[self.idx], not done_no_max)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def add_batch(self, obs, action, reward, next_obs, done, done_no_max):
        next_index = self.idx + self.window
        if next_index >= self.capacity:
            self.full = True
            maximum_index = self.capacity - self.idx
            np.copyto(self.obses[self.idx:self.capacity], obs[:maximum_index])
            np.copyto(self.actions[self.idx:self.capacity], action[:maximum_index])
            np.copyto(self.rewards[self.idx:self.capacity], reward[:maximum_index])
            np.copyto(self.next_obses[self.idx:self.capacity], next_obs[:maximum_index])
            np.copyto(self.not_dones[self.idx:self.capacity], done[:maximum_index] <= 0)
            np.copyto(self.not_dones_no_max[self.idx:self.capacity], done_no_max[:maximum_index] <= 0)
            remain = self.window - (maximum_index)
            if remain > 0:
                np.copyto(self.obses[0:remain], obs[maximum_index:])
                np.copyto(self.actions[0:remain], action[maximum_index:])
                np.copyto(self.rewards[0:remain], reward[maximum_index:])
                np.copyto(self.next_obses[0:remain], next_obs[maximum_index:])
                np.copyto(self.not_dones[0:remain], done[maximum_index:] <= 0)
                np.copyto(self.not_dones_no_max[0:remain], done_no_max[maximum_index:] <= 0)
            self.idx = remain
        else:
            np.copyto(self.obses[self.idx:next_index], obs)
            np.copyto(self.actions[self.idx:next_index], action)
            np.copyto(self.rewards[self.idx:next_index], reward)
            np.copyto(self.next_obses[self.idx:next_index], next_obs)
            np.copyto(self.not_dones[self.idx:next_index], done <= 0)
            np.copyto(self.not_dones_no_max[self.idx:next_index], done_no_max <= 0)
            self.idx = next_index

    def relabel_with_predictor(self, predictor):
        batch_size = 200
        total_iter = int(self.idx / batch_size)

        if self.idx > batch_size * total_iter:
            total_iter += 1

        for index in range(total_iter):
            last_index = (index + 1) * batch_size
            if (index + 1) * batch_size > self.idx:
                last_index = self.idx

            obses = self.obses[index * batch_size:last_index]
            actions = self.actions[index * batch_size:last_index]
            inputs = np.concatenate([obses, actions], axis=-1)

            pred_reward = predictor.r_hat_batch(inputs)
            self.rewards[index * batch_size:last_index] = pred_reward

    def sample_data(self, batch_size):

        idxs = np.random.randint(0, self.capacity if self.full else self.idx, size=batch_size) # curiosity exp idxs
        self.normal_exp_idx = self.normal_exp_idx_gen(idxs)
        obses = torch.as_tensor(self.obses[idxs], device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_obses = torch.as_tensor(self.next_obses[idxs], device=self.device).float()
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
        not_dones_no_max = torch.as_tensor(self.not_dones_no_max[idxs], device=self.device)

        return obses, actions, rewards, next_obses, not_dones, not_dones_no_max

    def sample(self, batch_size, step):

        # sample from replay buffer
        idxs = np.random.choice(self.normal_exp_idx, size=int(batch_size - batch_size * self.curiosity_exp_ratio), replace=True)
        obses = torch.as_tensor(self.obses[idxs], device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_obses = torch.as_tensor(self.next_obses[idxs], device=self.device).float()
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
        not_dones_no_max = torch.as_tensor(self.not_dones_no_max[idxs], device=self.device)
        # sample from intrinsic buffer
        idxs = np.random.randint(0, self.curiosity_buffer_size, size=int(batch_size * self.curiosity_exp_ratio))
        obses_int = torch.as_tensor(self.obses_int[idxs], device=self.device).float()
        actions_int = torch.as_tensor(self.actions_int[idxs], device=self.device)
        rewards_int = torch.as_tensor(self.rewards_int[idxs], device=self.device)
        next_obses_int = torch.as_tensor(self.next_obses_int[idxs], device=self.device).float()
        not_dones_int = torch.as_tensor(self.not_dones_int[idxs], device=self.device)
        not_dones_no_max_int = torch.as_tensor(self.not_dones_no_max_int[idxs], device=self.device)

        return torch.cat([obses, obses_int]), torch.cat([actions, actions_int]), torch.cat(
            [rewards, rewards_int]), torch.cat([next_obses, next_obses_int]), torch.cat(
            [not_dones, not_dones_int]), torch.cat([not_dones_no_max, not_dones_no_max_int])


    def sample_state_ent(self, batch_size):
        idxs = np.random.randint(0, self.capacity if self.full else self.idx, size=batch_size)

        obses = torch.as_tensor(self.obses[idxs], device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_obses = torch.as_tensor(self.next_obses[idxs], device=self.device).float()
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
        not_dones_no_max = torch.as_tensor(self.not_dones_no_max[idxs], device=self.device)

        if self.full:
            full_obs = self.obses
        else:
            full_obs = self.obses[: self.idx]
        full_obs = torch.as_tensor(full_obs, device=self.device)

        return obses, full_obs, actions, rewards, next_obses, not_dones, not_dones_no_max

    def update_curiosity_buffer(self, step):
        # update explore buffer (include replay buffer and preference buffer)
        obs, action, reward, next_obs, not_done, not_done_no_max = self.sample_data(self.curiosity_buffer_size)

        data_numpy = obs[:, :3].cpu().numpy()
        self.density_distribution = self.get_distribution(data_numpy)
        reward_intrinsic = torch.as_tensor(self.add_intrinsic_reward(data_numpy, step),
                                           device=self.device, dtype=torch.float32).view(-1, 1)
        self.rewards_int = reward + reward_intrinsic
        self.obses_int = obs
        self.next_obses_int = next_obs
        self.actions_int = action
        self.not_dones_int = not_done
        self.not_dones_no_max_int = not_done_no_max

    def update_preference_distribution(self, sa_t_1, sa_t_2, labels):
        # add preference data
        st_at = np.array([sa_t_1[i, :, :3] if labels[i] == 0 else sa_t_2[i, :, :3] for i in range(len(labels))]).reshape(-1, 3)
        next_idx = self.pre_buffer_idx + st_at.shape[0]
        if next_idx >= self.pre_buffer_capacity:
            self.pre_buffer_full = True

            end_len = self.pre_buffer_capacity - self.pre_buffer_idx
            np.copyto(self.obses_pre_buffer[self.pre_buffer_idx:self.pre_buffer_capacity], st_at[:end_len])

            start_len = st_at.shape[0] - end_len
            np.copyto(self.obses_pre_buffer[:start_len], st_at[end_len:])
            self.pre_buffer_idx = start_len
        else:
            np.copyto(self.obses_pre_buffer[self.pre_buffer_idx:next_idx], st_at)
            self.pre_buffer_idx = next_idx

        # update preference data distribution
        idx_prebuffer = np.random.randint(0, self.pre_buffer_capacity if self.pre_buffer_full else self.pre_buffer_idx,
                                          size=1000)

        self.pre_distribution = self.get_distribution(self.obses_pre_buffer[idx_prebuffer])


    def add_intrinsic_reward(self, obs_batch, step):
        point_batch = np.array(obs_batch).reshape(-1, 3)
        pre_score = np.exp(self.pre_distribution.score_samples(point_batch))  # high preference density
        density_score = 1 / (np.exp(self.density_distribution.score_samples(point_batch)))  # low replay buffer density
        intrinsic = pre_score * density_score
        intrinsic_scale = (intrinsic - np.min(intrinsic)) / (np.max(intrinsic) - np.min(intrinsic))
        weight = 0.1 * ((1 - self.time_de) ** int(step))

        return weight * intrinsic_scale

    def get_distribution(self, obs_batch):
        kde = KernelDensity(kernel='gaussian', bandwidth=1).fit(obs_batch)
        return kde

    def normal_exp_idx_gen(self, idxs):
        idx_buffer = np.arange(0, self.capacity if self.full else self.idx)
        normal_exp_idxs = np.setdiff1d(idx_buffer, idxs)
        return normal_exp_idxs
