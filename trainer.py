from logging import BufferingFormatter
import time
import ray
from models import Model
import copy
import torch
import numpy as np
import torch.nn as nn
import time
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

@ray.remote
class Trainer:
    def __init__(self,checkpoint,share_storage,replay_buffer) -> None:
        self.model=Model().cuda()
        self.model.set_weights(copy.deepcopy(checkpoint["weights"]))

        self.training_step=checkpoint['training_step']
        self.trained_step=checkpoint['max_training_step']
        self.mini_batch=checkpoint['mini_batch']
        self.model_save_iter=checkpoint['model_save_iter']
        self.entropy_coef=checkpoint['entropy_coef']
        self.value_loss_coef=checkpoint['value_loss_coef']
        self.len_episode=checkpoint['len_episode']
        self.gamma=checkpoint['gamma']
        self.num_reuse=checkpoint['num_reuse']
        self.epsilon=checkpoint['epsilon']
        self.gae_lambda=checkpoint['gae_lambda']
        self.num_sample=checkpoint['num_sample']
        self.share_storage=share_storage
        self.replay_buffer=replay_buffer
        self.learn_step_counter=1
        self.opt=torch.optim.Adam(self.model.parameters(),lr=checkpoint['lr'])
        self.model.train()
        self.test_pointer=0
        self.flag=True
        self.time_list1=[]
        self.time_list2=[]
        self.time_list3=[]
        print('trainer init done')


    def continous_update_weights(self):
        while not ray.get(self.share_storage.get_info.remote('start_training')):
            time.sleep(0.1)
        print('start training')
        traj1=self.replay_buffer[0].get_traj.remote()
        while True:
            # self.a = time.time()
            if self.flag:
                traj2 = self.replay_buffer[0].get_traj.remote()
                traj = ray.get(traj1)
                self.flag = not self.flag
            else:
                traj1 = self.replay_buffer[0].get_traj.remote()
                traj = ray.get(traj2)
                self.flag = not self.flag
            # self.b=time.time()
            # to cuda
            memory_obs,memory_act,memory_hx,memory_reward,memory_log_prob,memory_done=[item.cuda() for item in traj]

            with torch.no_grad():
                _,memory_value=self.model((memory_obs,memory_hx),memory_done)
                memory_value=torch.stack(memory_value)
                memory_returns,memory_adv=self.get_gae(memory_reward,memory_done,memory_value)
            
            a_loss,c_loss,entropy=self.update_weights(memory_obs,memory_hx,memory_act,memory_log_prob,memory_returns,memory_adv,memory_done)
            self.learn_step_counter=self.learn_step_counter%self.model_save_iter
            if self.learn_step_counter==0:
                
                # self.share_storage.save_checkpoint.remote()
                # print('---------update: ',a_loss,c_loss,entropy)
                self.share_storage.set_info.remote({"weights": copy.deepcopy(self.model.get_weights())})
            self.learn_step_counter+=1
            
            # print(self.b-self.a,self.c-self.b,self.d-self.c)


    def update_weights(self,memory_obs,memory_hx,memory_act,memory_log_prob,memory_returns,memory_adv,memory_done):

        for _ in range(self.num_reuse):
            data_generator=self.make_batch(memory_obs,memory_hx,memory_act,memory_log_prob,memory_returns,memory_adv,memory_done)
            for sample in data_generator:
                batch_obs,batch_hx,batch_action,batch_log_prob_old,batch_returns,batch_advantage,batch_done=sample
                # self.c=time.time()
                prob, value,= self.model((batch_obs,batch_hx),batch_done)
                # self.d=time.time()
                prob=torch.vstack(prob)
                value=torch.vstack(value)
                batch_log_prob_old=batch_log_prob_old.view(-1,1)
                batch_action=batch_action.long().view(-1,1)
                batch_advantage=batch_advantage.view(-1,1)
                batch_returns=batch_returns.view(-1,1)
                log_prob = torch.log(prob)
                entropy = -(log_prob*prob).sum(1, keepdim=True).mean()
                action_log_prob = log_prob.gather(1, batch_action)
                ratio = torch.exp(action_log_prob-batch_log_prob_old)
                surr1 = ratio*batch_advantage
                surr2 = torch.clamp(ratio, 1.0-self.epsilon, 1.0+self.epsilon)*batch_advantage
                a_loss = -torch.min(surr1, surr2).mean()

                c_loss=0.5*(batch_returns-value).pow(2).mean()

                self.opt.zero_grad()
                (a_loss+c_loss*self.value_loss_coef-entropy*self.entropy_coef).backward()
                nn.utils.clip_grad_norm_(self.model.parameters(),0.5)
                self.opt.step()

        self.test_pointer+=1
        return a_loss.item(),c_loss.item(),entropy.item()

    def get_gae(self, memory_reward, memory_done, memory_value):
        gae = torch.zeros(self.num_sample, self.len_episode+1, 1).cuda()
        memory_returns = torch.zeros(self.num_sample, self.len_episode, 1).cuda()
        roll_memory=torch.roll(memory_done, -1)[:, :-1]
        roll_value=torch.roll(memory_value, -1)[:, :-1]
        delta = memory_reward + \
            roll_value * self.gamma * (1-roll_memory) \
            -memory_value[:, :-1]

        for i in reversed(range(self.len_episode)):
            gae[:, i] = delta[:, i]+self.gamma * self.gae_lambda*(1-roll_memory[:, i])*gae[:, i+1]

        memory_value = memory_value[:,:-1]
        memory_returns=gae[:,:-1]+memory_value
        memory_adv = memory_returns-memory_value
        memory_adv = (memory_adv-memory_adv.mean())/(memory_adv.std()+1e-5)
        return memory_returns, memory_adv


    def make_batch(self,memory_obs,memory_hx,memory_act,memory_log_prob,memory_returns,memory_adv,memory_done):

        num_sample=memory_obs.size(0)
        mini_batch=num_sample//self.mini_batch
        batch_obs=[]
        batch_hx=[]
        batch_action=[]
        batch_log_prob_old=[]
        batch_returns=[]
        batch_advantage=[]
        batch_done=[]
        sampler = BatchSampler(SubsetRandomSampler(range(num_sample)),mini_batch,drop_last=True)
        for idx in sampler:
            batch_obs=memory_obs[idx,:-1]
            batch_hx=memory_hx[idx]
            batch_action=memory_act[idx]
            batch_log_prob_old=memory_log_prob[idx]
            batch_returns=memory_returns[idx]
            batch_advantage=memory_adv[idx]
            batch_done=memory_done[idx,:-1]
        
        
            yield batch_obs,batch_hx,batch_action,batch_log_prob_old,batch_returns,batch_advantage,batch_done

