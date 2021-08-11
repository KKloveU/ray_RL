import copy
import ray
import random
import numpy as np
import torch

@ray.remote
class ReplayBuffer:
    def __init__(self, checkpoint, share_storage) -> None:
        self.memory_obs = []
        self.memory_value = []
        self.memory_action = []
        self.memory_reward = []
        self.memory_log_prob = []
        self.memory_done = []
        self.memory_returns = []
        self.memory_advantage = []
        self.index = 0
        self.share_storage = share_storage
        self.start_training = checkpoint['start_training']
        self.memory_size = checkpoint['memory_size']
        self.batch_size = checkpoint['batch_size']

        self.minibatch_size=4
        self.pointer=0
    def get_batch(self):
        batch_obs, batch_value, batch_action, batch_log_prob, batch_returns, batch_advantage,  batch_done = [], [], [], [], [], [], []

        # idx = [i for i in range(len(self.memory_reward))]
        # random.shuffle(idx)
        # for i in range(0,len()):

        for _ in range(self.batch_size):
            idx = random.randint(0, len(self.memory_reward) - 1)
            batch_obs.append(self.memory_obs[idx])
            batch_value.append(self.memory_value[idx])
            batch_action.append(self.memory_action[idx])
            batch_log_prob.append(self.memory_log_prob[idx])
            batch_returns.append(self.memory_returns[idx])
            batch_advantage.append(self.memory_advantage[idx])
            batch_done.append(self.memory_done[idx])
        return np.stack(batch_obs), np.vstack(batch_value), np.vstack(batch_action), np.vstack(batch_log_prob), np.vstack(batch_returns), np.vstack(batch_advantage), np.vstack(batch_done)

    def upload_memory(self, game_history):
        self.memory_obs += game_history.obs_history
        self.memory_value += game_history.value_history
        self.memory_action += game_history.action_history
        self.memory_reward += game_history.reward_history
        self.memory_log_prob += game_history.log_prob_history
        self.memory_done +=game_history.done_history
        self.memory_returns += game_history.returns_history
        self.memory_advantage += game_history.advantage_history

        # if len(self.memory_obs) > self.memory_size:
        #     self.memory_obs = self.memory_obs[-int(self.memory_size):]
        #     self.memory_action = self.memory_action[-int(self.memory_size):]
        #     self.memory_reward = self.memory_reward[-int(self.memory_size):]
        #     self.memory_obs_ = self.memory_obs_[-int(self.memory_size):]
        #     self.memory_vtarget = self.memory_vtarget[-int(self.memory_size):]
        #     self.memory_done = self.memory_done[-int(self.memory_size):]
        #     self.share_storage.set_info.remote({"start_training": True})

    def clear_memory(self):
        self.memory_obs = []
        self.memory_value = []
        self.memory_action = []
        self.memory_reward = []
        self.memory_log_prob = []
        self.memory_done = []
        self.memory_returns = []
        self.memory_advantage = []
