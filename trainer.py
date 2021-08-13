from math import ceil
import time
import ray
from models import Model
import copy
import torch
import numpy as np
import torch.nn as nn
import time

@ray.remote
class Trainer:
    def __init__(self,checkpoint,share_storage,replay_buffer) -> None:
        self.model=Model().cuda()
        self.model.set_weights(copy.deepcopy(checkpoint["weights"]))

        self.training_step=checkpoint['training_step']
        self.trained_step=checkpoint['max_training_step']
        self.batch_size=checkpoint['batch_size']
        self.model_save_iter=checkpoint['model_save_iter']
        self.entropy_coef=checkpoint['entropy_coef']
        self.value_loss_coef=checkpoint['value_loss_coef']
        self.memory_update_iter=checkpoint['memory_update_iter']
        self.gamma=checkpoint['gamma']
        self.epsilon=checkpoint['epsilon']
        self.gae_lambda=checkpoint['gae_lambda']
        self.share_storage=share_storage
        self.replay_buffer=replay_buffer
        self.learn_step_counter=1
        self.opt=torch.optim.Adam(self.model.parameters(),lr=checkpoint['lr'])
        self.model.train()
        self.test_pointer=0
        self.flag=True
        
        print('trainer init done')


    def continous_update_weights(self):
        while not ray.get(self.share_storage.get_info.remote('start_training')):
            time.sleep(0.1)
        print('start training')
        
        while True:
            traj=ray.get(self.replay_buffer.get_traj.remote())
            traj.cuda()
            with torch.no_grad():
                _,memory_value=self.model(traj.obs_history)
            memory_value,memory_returns,memory_adv=self.get_gae(traj,memory_value)
            # print(memory_returns)
            # for _ in range(3):
            a_loss,c_loss,entropy=self.update_weights(traj,memory_returns,memory_adv)

            self.learn_step_counter=self.learn_step_counter%self.model_save_iter
            if self.learn_step_counter==0:
                # self.share_storage.save_checkpoint.remote()
                print('update: ',self.learn_step_counter,a_loss,c_loss,entropy)
                self.share_storage.set_info.remote({"weights": copy.deepcopy(self.model.get_weights())})
            self.learn_step_counter+=1


    def update_weights(self,traj,memory_returns,memory_adv):
        
        batch_obs=traj.obs_history[:-1]
        batch_value=traj.value_history
        batch_action=traj.action_history.long()
        batch_log_prob_old=traj.log_prob_history
        batch_returns=memory_returns
        batch_advantage=memory_adv

        for _ in range(3):

            prob,value=self.model(batch_obs)
            log_prob=torch.log(prob)
            entropy=-(log_prob*prob).sum(1, keepdim=True).mean()

            action_log_prob=log_prob.gather(1,batch_action)

            ratio=torch.exp(action_log_prob-batch_log_prob_old)
            surr1 = ratio*batch_advantage
            surr2=torch.clamp(ratio,1.0-self.epsilon,1.0+self.epsilon)*batch_advantage
            a_loss=-torch.min(surr1,surr2).mean()

            # value_clipped=batch_value+(value-batch_value).clamp(-self.epsilon,self.epsilon)
            # c_loss=(value-batch_returns).pow(2)
            # c_loss_clip=(value_clipped-batch_returns).pow(2)
            # c_loss=0.5*torch.max(c_loss,c_loss_clip).mean()

            c_loss=0.5*(batch_returns-value).pow(2).mean()

            self.opt.zero_grad()
            (a_loss+c_loss*self.value_loss_coef-entropy*self.entropy_coef).backward()
            # (a_loss+c_loss*0.5).backward()
            nn.utils.clip_grad_norm_(self.model.parameters(),0.5)
            self.opt.step()

        self.test_pointer+=1
        return a_loss.item(),c_loss.item(),entropy.item()

    def get_gae(self, traj, memory_value):
        gae = torch.zeros(1).cuda()
        memory_returns = torch.zeros(self.memory_update_iter, 1).cuda()
        for i in reversed(range(self.memory_update_iter)):
            delta = traj.reward_history[i]+memory_value[i+1] * self.gamma*(1-traj.done_history[i])-memory_value[i]
            gae = delta+self.gamma*self.gae_lambda*(1-traj.done_history[i])*gae
            memory_returns[i] = gae+memory_value[i]
        memory_value = memory_value[:-1]
        memory_adv = memory_returns-memory_value
        memory_adv = (memory_adv-memory_adv.mean())/(memory_adv.std()+1e-7)
        return memory_value, memory_returns, memory_adv
