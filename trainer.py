from math import ceil
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
        self.batch_size=checkpoint['batch_size']
        self.model_save_iter=checkpoint['model_save_iter']
        self.entropy_coef=checkpoint['entropy_coef']
        self.value_loss_coef=checkpoint['value_loss_coef']
        self.len_episode=checkpoint['len_episode']
        self.gamma=checkpoint['gamma']
        self.num_reuse=checkpoint['num_reuse']
        self.epsilon=checkpoint['epsilon']
        self.gae_lambda=checkpoint['gae_lambda']
        self.num_workers=checkpoint['num_workers']
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
            self.a = time.time()
            if self.flag:
                traj2 = self.replay_buffer[0].get_traj.remote()
                traj = ray.get(traj1)
                self.flag = not self.flag
            else:
                traj1 = self.replay_buffer[0].get_traj.remote()
                traj = ray.get(traj2)
                self.flag = not self.flag

            # to cuda
            memory_obs,memory_act,memory_reward,memory_log_prob,memory_done=[item.cuda() for item in traj]


            with torch.no_grad():
                memory_value=[]
                for i in range(self.num_workers):
                    _,value=self.model(memory_obs[i])    
                    memory_value.append(value)
                memory_value=torch.stack(memory_value).cuda()
                memory_returns,memory_adv=self.get_gae(memory_reward,memory_done,memory_value)
            

            a_loss,c_loss,entropy=self.update_weights(memory_obs,memory_act,memory_log_prob,memory_returns,memory_adv)
            self.learn_step_counter=self.learn_step_counter%self.model_save_iter
            if self.learn_step_counter==0:
                # self.share_storage.save_checkpoint.remote()
                print('---------update: ',a_loss,c_loss,entropy)
                self.share_storage.set_info.remote({"weights": copy.deepcopy(self.model.get_weights())})
                self.time_list1=[]
                self.time_list2=[]
                self.time_list3=[]
            self.learn_step_counter+=1


    def update_weights(self,memory_obs,memory_act,memory_log_prob,memory_returns,memory_adv):

        for _ in range(self.num_reuse):
            data_generator=self.make_batch(memory_obs,memory_act,memory_log_prob,memory_returns,memory_adv)
            
            for batch in data_generator:
                batch_obs, batch_action, batch_log_prob_old, batch_returns, batch_advantage = batch

                prob, value = self.model(batch_obs)
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


    def get_gae(self, memory_reward,memory_done, memory_value):        
        gae = torch.zeros(self.num_workers,self.len_episode+1,1).cuda()
        memory_returns = torch.zeros(self.num_workers,self.len_episode, 1).cuda()

        delta=memory_reward+torch.roll(memory_value,-1)[:,:-1]*self.gamma*(1-memory_done)-memory_value[:,:-1]

        for i in reversed(range(self.len_episode)):
            gae[:,i] = delta[:,i]+self.gamma*self.gae_lambda*(1-memory_done[:,i])*gae[:,i+1]

        memory_value = memory_value[:,:-1]
        memory_returns=gae[:,:-1]+memory_value
        memory_adv = memory_returns-memory_value
        memory_adv = (memory_adv-memory_adv.mean())/(memory_adv.std()+1e-5)
        return memory_returns, memory_adv


    def make_batch(self,memory_obs,memory_act,memory_log_prob,memory_returns,memory_adv):
        memory_obs=memory_obs[:,:-1].reshape(-1,4,84,84)
        memory_act=memory_act.view(-1,1)
        memory_log_prob=memory_log_prob.view(-1,1)
        memory_returns=memory_returns.view(-1,1)
        memory_adv=memory_adv.view(-1,1)

        sampler = BatchSampler(SubsetRandomSampler(range(self.len_episode*self.num_workers)),self.batch_size,True)
        for index in sampler:
            batch_obs=memory_obs[index]
            batch_action=memory_act.long()[index]
            batch_log_prob_old=memory_log_prob[index]
            batch_returns=memory_returns[index]
            batch_advantage=memory_adv[index]
            yield batch_obs,batch_action,batch_log_prob_old,batch_returns,batch_advantage

