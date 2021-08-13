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
        self.batch_size = checkpoint['batch_size']


    def get_traj(self):
        idx = random.randint(0,int(self.memory_size)-1)
        return self.memory[idx]


    def upload_memory(self, game_history):
        self.memory.append(game_history)

        if len(self.memory) > self.memory_size:
            self.memory = self.memory[-int(self.memory_size):]
            self.share_storage.set_info.remote({"start_training": True})


    def clear_memory(self):
        self.memory=[]
