import time
import ray
import cv2
import models
import copy
import torch
import numpy as np
import torch.nn as nn
import time


@ray.remote
class Trainer:
    def __init__(self,checkpoint,share_storage) -> None:
        self.model=models.Model().cuda()
        self.model.set_weights(copy.deepcopy(checkpoint["weights"]))
        self.optimizer=torch.optim.Adam(self.model.parameters(),lr=checkpoint['lr'])
        self.share_storage=share_storage
        
        self.training_step=checkpoint['training_step']
        self.trained_step=checkpoint['max_training_step']
        self.tau=checkpoint["tau"]
        self.batch_size=checkpoint['batch_size']
        self.gamma=checkpoint['gamma']
        self.model_save_iter=checkpoint['model_save_iter']
        self.memory_update_iter=checkpoint['memory_update_iter']

        self.learn_step_counter=1
        print('trainer init done')
        
    def continous_update_weights(self):
        pass

    def update_weights(self,game_history):
        s,a,v_t=game_history.obs_history,game_history.action_history,game_history.v_target_history
        loss=self.loss_func(s,a,v_t)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.share_storage.set_info.remote({"weights": copy.deepcopy(self.model.get_weights())})
        self.learn_step_counter=self.learn_step_counter%self.model_save_iter
        if self.learn_step_counter==0:
            self.share_storage.save_checkpoint.remote()
        self.learn_step_counter+=1

    def loss_func(self, s, a, v_t):
        probs, values = self.model.forward(s)
        td = v_t - values
        c_loss = td.pow(2)
        m = self.model.distribution(probs)
        exp_v = m.log_prob(a) * td.detach().squeeze()
        a_loss = -exp_v
        total_loss = (c_loss + a_loss).mean()
        return total_loss