from math import ceil
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
    def __init__(self,checkpoint,share_storage,replay_buffer) -> None:
        self.actor_eval=models.Actor().cuda()
        self.actor_target=models.Actor().cuda()
        self.critic=models.Critic().cuda()
        self.actor_eval.set_weights(copy.deepcopy(checkpoint["actor_weights"]))
        self.actor_target.set_weights(copy.deepcopy(checkpoint["actor_weights"]))
        self.critic.set_weights(copy.deepcopy(checkpoint["critic_weights"]))

        self.training_step=checkpoint['training_step']
        self.trained_step=checkpoint['max_training_step']
        self.batch_size=checkpoint['batch_size']
        self.gamma=checkpoint['gamma']
        self.replace_target_iter=checkpoint['replace_target_iter']
        self.entropy_beta=checkpoint['entropy_beta']
        self.num_update=checkpoint['num_update']
        self.epsilon=checkpoint['epsilon']
        self.share_storage=share_storage
        self.replay_buffer=replay_buffer

        self.learn_step_counter=1

        self.c_opt=torch.optim.Adam(self.critic.parameters(),lr=checkpoint['lr'])
        self.a_opt=torch.optim.Adam(self.actor_eval.parameters(),lr=checkpoint['lr'])

        self.flag=True
        print('trainer init done')

    def continous_update_weights(self):
        while True:
            while not ray.get(self.share_storage.get_info.remote('start_training')):
                time.sleep(0.1)

            batch=self.replay_buffer.get_batch.remote()
            while True:
                if self.flag:
                    batch_=self.replay_buffer.get_batch.remote()
                    batch=ray.get(batch)
                    self.update_weights(batch)
                    self.flag = not self.flag

                else:
                    batch=self.replay_buffer.get_batch.remote()
                    batch_=ray.get(batch_)
                    self.update_weights(batch_)
                    self.flag = not self.flag

                self.learn_step_counter=self.learn_step_counter%self.replace_target_iter
                if self.learn_step_counter==0:
                    self.share_storage.save_checkpoint.remote()
                
                self.share_storage.set_info.remote({"actor_weights": copy.deepcopy(self.actor_eval.get_weights())})
                self.share_storage.set_info.remote({"critic_weights": copy.deepcopy(self.critic.get_weights())})
                self.learn_step_counter+=1


    def update_weights(self,batch):
        s,a,v_t=copy.deepcopy(batch)
        s=torch.FloatTensor(s).cuda().permute(0,3,1,2)
        a=torch.LongTensor(a).cuda()
        v_t=torch.FloatTensor(v_t).cuda()
        self.actor_target.load_state_dict(self.actor_eval.get_weights())

        '''update critic'''
        for _ in range(self.num_update):
            values = self.critic(s)
            td = v_t - values
            c_loss = td.pow(2).mean()

            self.c_opt.zero_grad()
            c_loss.backward()
            self.c_opt.step()
            # for param in self.critic.parameters():
            #     param.grad.data.clamp_(-1,1)
            # self.c_opt.step()
        
        values = self.critic(s)
        
        for _ in range(self.num_update):
            prob=self.actor_eval(s)
            prob_=self.actor_target(s)

            ''''''
            log_prob=torch.log(prob.gather(1,a.long()))
            log_prob_=torch.log(prob_.gather(1,a.long()))
            ratio=torch.exp(log_prob-log_prob_)
            td=v_t-values
            surr1=ratio*td.detach()
            surr2=torch.clamp(ratio,1.0-self.epsilon,1.0+self.epsilon)*td.detach()
            a_loss=-torch.min(surr1,surr2).mean()
            ''''''
            # ratio=prob/(prob_.detach()+1e5)
            # td=v_t-values
            # surr=ratio*td.detach()
            # a_loss=-(torch.min(surr,torch.clip(ratio,1-self.epsilon,1+self.epsilon)*td.detach())).mean()

            self.a_opt.zero_grad()
            a_loss.backward()
            # for param in self.actor_eval.parameters():
            #     param.grad.data.clamp_(-1,1)
            self.a_opt.step()


    #         # log_prob=torch.log(probs).gather(1,a)
    #         # exp_v=log_prob*td.detach()

    #         # entropy=probs*torch.log(probs)
    #         # entropy=-torch.sum(probs*torch.log(probs),dim=1,keepdim=True)
    #         # # print(entropy)
    #         # exp_v=self.entropy_beta*entropy.detach()+exp_v

    #         # a_loss = -exp_v
                




    #     # print(v_t)
    #     a_loss,c_loss=self.loss_func(s,a,v_t)
        
    #     self.a_opt.zero_grad()
    #     a_loss.backward()
    #     for param in self.actor.parameters():
    #         param.grad.data.clamp_(-1,1)
    #     self.a_opt.step()

    #     self.c_opt.zero_grad()
    #     c_loss.backward()
    #     for param in self.critic.parameters():
    #         param.grad.data.clamp_(-1,1)
    #     self.c_opt.step()

    # def loss_func(self, s, a, v_t):
    #     probs = self.actor(s)
    #     values = self.critic(s)
        
    #     td = v_t - values
    #     c_loss = td.pow(2)

    #     # print(td)
    #     # m = self.actor.distribution(probs)

    #     log_prob=torch.log(probs).gather(1,a)
    #     exp_v=log_prob*td.detach()
    #     # print(m)
    #     # exp_v = m.log_prob(a) * td.detach().squeeze()
    #     entropy=probs*torch.log(probs)
    #     entropy=-torch.sum(probs*torch.log(probs),dim=1,keepdim=True)
    #     # print(entropy)
    #     exp_v=self.entropy_beta*entropy.detach()+exp_v

    #     a_loss = -exp_v

    #     # total_loss = (c_loss + a_loss).mean()
    #     return a_loss.mean(),c_loss.mean()