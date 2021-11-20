import copy
import ray
import random
import numpy as np
import torch

@ray.remote
class ReplayBuffer:
    def __init__(self, checkpoint, share_storage) -> None:
        self.memory=[]
        self.share_storage = share_storage
        self.memory_size = checkpoint['memory_size']
        self.num_sample = checkpoint['num_sample']


    # def get_traj(self):
    #     idx = random.randint(0,int(self.memory_size)-1)
    #     return copy.deepcopy(self.memory[idx])
        
    def get_traj(self):
        batch_obs=[]
        batch_hx=[]
        batch_act=[]
        batch_reward=[]
        batch_log_prob=[]
        batch_done=[]

        for i in range(self.num_sample):
            idx = random.randint(0,int(self.memory_size)-1)
            batch_obs.append(self.memory[idx].obs_history)
            batch_hx.append(self.memory[idx].hx_history)
            batch_act.append(self.memory[idx].action_history)
            batch_reward.append(self.memory[idx].reward_history)
            batch_log_prob.append(self.memory[idx].log_prob_history)
            batch_done.append(self.memory[idx].done_history)
        return (torch.stack(batch_obs),torch.stack(batch_act),torch.stack(batch_hx),torch.stack(batch_reward),torch.stack(batch_log_prob),torch.stack(batch_done))


    def upload_memory(self, game_history):
        self.memory.append(game_history)

        if len(self.memory) > self.memory_size:
            self.memory = self.memory[-int(self.memory_size):]
            self.share_storage.set_info.remote({"start_training": True})


    def clear_memory(self):
        self.memory=[]
