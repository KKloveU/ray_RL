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
    def __init__(self,checkpoint,replay_buffer,share_storage) -> None:
        # self.eval_model=nn.DataParallel(models.Model())       #multi-GPU
        # self.target_model=nn.DataParallel(models.Model())     #multi-GPU
        self.eval_model=models.Model()
        self.target_model=models.Model()
        # self.eval_model.module.set_weights(copy.deepcopy(checkpoint["weights"]))      #multi-GPU
        # self.target_model.module.set_weights(copy.deepcopy(checkpoint["weights"]))    #multi-GPU
        self.eval_model.set_weights(copy.deepcopy(checkpoint["weights"]))
        self.target_model.set_weights(copy.deepcopy(checkpoint["weights"]))

        self.eval_model.cuda()
        self.target_model.cuda()

        self.replay_buffer=replay_buffer
        self.share_storage=share_storage
        self.gamma=checkpoint['gamma']
        self.tau=checkpoint["tau"]
        self.batch_size=checkpoint['batch_size']
        self.training_step=checkpoint['training_step']
        self.trained_step=checkpoint['max_training_step']
        self.replace_target_iter=checkpoint['replace_target_iter']

        self.flag=True
        self.learn_step_counter=1
        self.loss_fn=torch.nn.SmoothL1Loss(reduction="none")
        self.optimizer=torch.optim.Adam(self.eval_model.parameters(),lr=checkpoint['lr'])
        print('trainer init done')
        
    def continous_update_weights(self):
        print('wait train')
        while not ray.get(self.share_storage.get_info.remote('start_training')):
            time.sleep(0.1)
        print('start train-----------------------------------------------------')
        
        batch=self.replay_buffer.get_batch.remote(self.batch_size)
        while self.learn_step_counter<self.max_training_step:
            if self.flag:
                batch_=self.replay_buffer.get_batch.remote(self.batch_size)
                batch=ray.get(batch)
                tree_idx,abs_error=self.update_weights(batch)
                self.flag = not self.flag

            else:
                batch=self.replay_buffer.get_batch.remote(self.batch_size)
                batch_=ray.get(batch_)
                tree_idx,abs_error=self.update_weights(batch_)
                self.flag = not self.flag
            self.replay_buffer.batch_update.remote(tree_idx,abs_error)

            self.learn_step_counter=self.learn_step_counter%self.replace_target_iter
            if self.learn_step_counter==0:
                # self.share_storage.set_info.remote({"weights": copy.deepcopy(self.eval_model.module.get_weights())})
                self.share_storage.set_info.remote({"weights": copy.deepcopy(self.eval_model.get_weights())})
                self.share_storage.save_checkpoint.remote()
                print('net_replace!!!!')

            for target_param, param in zip(self.target_model.parameters(), self.eval_model.parameters()):
                target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)

            self.learn_step_counter+=1

    def update_weights(self,batch):
        tree_idx, batch_obs, batch_act, batch_reward, batch_obs_, batch_done, ISWeights=copy.deepcopy(batch)
        batch_obs=torch.FloatTensor(np.stack(batch_obs)).permute(0,3,1,2).cuda()
        batch_act=torch.LongTensor(np.vstack(batch_act)).cuda()
        batch_reward=torch.FloatTensor(np.vstack(batch_reward)).cuda()
        batch_obs_=torch.FloatTensor(np.stack(batch_obs_)).permute(0,3,1,2).cuda()
        batch_done=torch.BoolTensor(np.vstack(batch_done)).cuda()
        batch_weight=torch.FloatTensor(ISWeights).cuda()

        q_eval=self.eval_model(batch_obs).gather(1,batch_act)
        eval_act_index=self.eval_model(batch_obs_).max(1)[1]
        q_next=self.target_model(batch_obs_)
        q_target=batch_reward+self.gamma*q_next[list(range(self.batch_size)),eval_act_index].view(-1,1)
        q_target=torch.where(batch_done,batch_reward,q_target)

        loss = (batch_weight * self.loss_fn(q_eval,q_target.detach())).mean()
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.eval_model.parameters():                
            param.grad.data.clamp_(-1, 1) 
        self.optimizer.step()

        abs_error=torch.abs(q_eval-q_target).detach().cpu().numpy()
        return tree_idx,abs_error


        

        