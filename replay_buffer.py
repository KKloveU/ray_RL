import copy
from types import DynamicClassAttribute
import ray
import random
import numpy as np

@ray.remote
class ReplayBuffer:
    def __init__(self,checkpoint,share_storage) -> None:
        self.memory_obs=[]
        self.memory_act=[]
        self.memory_reward=[]
        self.memory_obs_=[]
        self.memory=[]
        self.index=0
        self.share_storage=share_storage
        self.start_training=checkpoint['start_training']
        self.memory_size=checkpoint['memory_size']
        self.batch_size=checkpoint['batch_size']
        
    def get_batch(self):
        # sample_index=np.random.choice(int(self.memory_size),int(self.batch_size))
        # batch_obs=self.memory_obs[sample_index]
        # batch_act=self.memory_act[sample_index]
        # batch_reward=self.memory_reward[sample_index]
        # batch_obs_=self.memory_obs_[sample_index]
        # return np.stack(batch_obs),np.vstack(batch_act),np.vstack(batch_reward),np.stack(batch_obs_)

        batch_obs, batch_act, batch_reward, batch_obs_= [], [], [], []
        for _ in range(self.batch_size):
            idx = random.randint(0, self.memory_size - 1)
            batch_obs.append(self.memory_obs[idx])
            batch_act.append(self.memory_act[idx])
            batch_reward.append(self.memory_reward[idx])
            batch_obs_.append(self.memory_obs_[idx])
        return np.stack(batch_obs),np.vstack(batch_act),np.vstack(batch_reward),np.stack(batch_obs_)

        # batch_obs, batch_act, batch_reward, batch_obs_= [], [], [], []
        # for _ in range(self.batch_size):
        #     idx = random.randint(0, self.memory_size - 1)
        #     frame, action, reward, next_frame= self.memory[idx]
        #     batch_obs.append(frame)
        #     batch_act.append(action)
        #     batch_reward.append(reward)
        #     batch_obs_.append(next_frame)
        # return np.stack(batch_obs),np.vstack(batch_act),np.vstack(batch_reward),np.stack(batch_obs_)
    
    # def store_transition(self,obs,act,reward,obs_):
    #     if(len(self.memory)<self.memory_size):
    #         self.memory.append([obs,act,reward,obs_])
    #     else: # buffer is full
    #         self.share_storage.set_info.remote({"start_training":True})
    #         self.memory[int(self.index)] = [obs,act,reward,obs_]
    #     self.index = (self.index + 1) % self.memory_size


    def store_memory(self,game_history):
        self.memory_obs+=game_history.obs_history
        self.memory_act+=game_history.act_history
        self.memory_reward+=game_history.reward_history
        self.memory_obs_+=game_history.obs_next_history
        if len(self.memory_obs)>self.memory_size:
            self.memory_obs=self.memory_obs[-int(self.memory_size):]
            self.memory_act=self.memory_act[-int(self.memory_size):]
            self.memory_reward=self.memory_reward[-int(self.memory_size):]
            self.memory_obs_=self.memory_obs_[-int(self.memory_size):]
            self.share_storage.set_info.remote({"start_training":True})

