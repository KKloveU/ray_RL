from math import ceil
from os import PRIO_USER
import time
import ray
import cv2
from ray.ray_constants import MONITOR_DIED_ERROR
from torch.nn.utils import clip_grad
from models import Model
import copy
import torch
import numpy as np
import torch.nn as nn
import time
import itertools

@ray.remote
class Trainer:
    def __init__(self,checkpoint,share_storage,replay_buffer) -> None:
        self.model=Model().cuda()
        self.model.set_weights(copy.deepcopy(checkpoint["weights"]))
        # self.target_model=Model().cuda()
        # self.target_model.set_weights(copy.deepcopy(checkpoint["weights"]))

        self.training_step=checkpoint['training_step']
        self.trained_step=checkpoint['max_training_step']
        self.batch_size=checkpoint['batch_size']
        self.gamma=checkpoint['gamma']
        self.model_save_iter=checkpoint['model_save_iter']
        self.entropy_beta=checkpoint['entropy_beta']
        self.epsilon=checkpoint['epsilon']
        self.share_storage=share_storage
        self.replay_buffer=replay_buffer
        self.learn_step_counter=1
        # self.opt = torch.optim.Adam([
            # {'params': itertools.chain(self.model.features.parameters(), self.model.actor.parameters()), 'lr': checkpoint['lr_a']},
            # {'params': itertools.chain(self.model.features.parameters(), self.model.value.parameters()), 'lr': checkpoint['lr_c']}])
        self.opt=torch.optim.Adam(self.model.parameters(),lr=checkpoint['lr'])
        self.model.train()
        self.test_pointer=0

        self.flag=True
        print('trainer init done')

    # def continous_update_weights(self):
    #     while not ray.get(self.share_storage.get_info.remote('start_training')):
    #         time.sleep(0.1)
    #     print('start training')
        
    #     batch=self.replay_buffer.get_batch.remote()
    #     while True:
    #         # if self.flag:
    #         #     batch_=self.replay_buffer.get_batch.remote()
    #         #     batch=ray.get(batch)
    #         #     self.update_weights(batch)
    #         #     self.flag = not self.flag

    #         # else:
    #         #     batch=self.replay_buffer.get_batch.remote()
    #         #     batch_=ray.get(batch_)
    #         #     self.update_weights(batch_)
    #         #     self.flag = not self.flag
            
    #         batch=ray.get(self.replay_buffer.get_batch.remote())
    #         self.update_weights(batch)

    #         self.learn_step_counter=self.learn_step_counter%self.model_save_iter
    #         if self.learn_step_counter==0:
    #             # self.share_storage.save_checkpoint.remote()
    #             self.share_storage.set_info.remote({"weights": copy.deepcopy(self.model.get_weights())})
    #         self.learn_step_counter+=1


    # def update_weights(self,batch):
    #     s,a,r,s_,v_t,done=copy.deepcopy(batch)
    #     s=torch.FloatTensor(s).cuda()
    #     a=torch.LongTensor(a).cuda()
    #     r=torch.FloatTensor(r).cuda()
    #     s_=torch.FloatTensor(s_).cuda()
    #     v_t=torch.FloatTensor(v_t).cuda()
    #     done=torch.LongTensor(done).cuda()

    #     prob,value=self.model(s)

    #     td_error=v_t-value
    #     c_loss=0.5*td_error.pow(2)
    #     # prob,td_error,adv=self.get_gae(s,r,s_,done)

    #     # c_loss=td_error.pow(2)

    #     log_prob=torch.log(prob).gather(1,a)
    #     a_loss=-log_prob*td_error.detach()

    #     if self.test_pointer%500==0:
    #         # print(a)
    #         print(a_loss)
    #         print('c_loss',c_loss.mean())
    #         print('a_loss',a_loss.mean())
    #         # print(c_loss.mean(),a_loss.mean())
        
    #     self.opt.zero_grad()
    #     (a_loss.sum()+c_loss.sum()).backward()
    #     torch.nn.utils.clip_grad_norm_(self.model.parameters(),40)
    #     self.opt.step()
    #     self.test_pointer+=1


    def update(self):
        
        for i in range(3):
            for ii in range(4):
                batch=ray.get(self.replay_buffer.get_batch.remote())
                batch_obs,batch_value,batch_action,batch_log_prob_old,batch_returns,batch_advantage,batch_done=copy.deepcopy(batch)

                batch_obs=torch.FloatTensor(batch_obs).cuda()
                batch_value=torch.FloatTensor(batch_value).cuda()
                batch_action=torch.LongTensor(batch_action).cuda()
                batch_log_prob_old=torch.FloatTensor(batch_log_prob_old).cuda()
                batch_returns=torch.FloatTensor(batch_returns).cuda()
                batch_advantage=torch.FloatTensor(batch_advantage).cuda()
                batch_done=torch.LongTensor(batch_done).cuda()
                # print(batch_obs.shape)
                # print('--------------')
                # print(batch_value)
                # print(batch_log_prob_old)
                # print('------------')
                prob,value=self.model(batch_obs)
                # print(batch_log_prob_old)
                log_prob=torch.log(prob)
                entropy=-(log_prob*prob).sum(1, keepdim=True).mean()

                action_log_prob=log_prob.gather(1,batch_action)

                ratio=torch.exp(action_log_prob-batch_log_prob_old)
                surr1 = ratio*batch_advantage
                surr2=torch.clamp(ratio,1.0-self.epsilon,1.0+self.epsilon)*batch_advantage
                a_loss=-torch.min(surr1,surr2).mean()

                value_clipped=batch_value+(value-batch_value).clamp(-self.epsilon,self.epsilon)
                c_loss=(value-batch_returns).pow(2)
                c_loss_clip=(value_clipped-batch_returns).pow(2)
                c_loss=0.5*torch.max(c_loss,c_loss_clip).mean()

                # c_loss=0.5*(batch_returns-value).pow(2).mean()

                self.opt.zero_grad()
                (a_loss+c_loss*0.5-entropy*0.01).backward()
                # (a_loss+c_loss*0.5).backward()
                nn.utils.clip_grad_norm_(self.model.parameters(),0.5)
                self.opt.step()

        self.replay_buffer.clear_memory.remote()
        # print('update:',self.test_pointer,'a_loss: ',a_loss.item(),' c_loss: ',c_loss.item())
        print('update:',self.test_pointer,'a_loss: ',a_loss.item(),' c_loss: ',c_loss.item(),' entropy: ',entropy.item())
        self.test_pointer+=1
        return self.model.get_weights()


    # def upload_memory(self,game_history):
    #     game_history=copy.deepcopy(game_history)
    #     s=torch.FloatTensor(game_history.obs_history).cuda()
    #     a=torch.LongTensor(game_history.action_history).unsqueeze(0).cuda()
    #     r=torch.FloatTensor(game_history.reward_history).cuda()
    #     s_=torch.FloatTensor(game_history.obs_next_history).cuda()
    #     v_t=torch.FloatTensor(game_history.vtarget_history).cuda()
    #     done=torch.LongTensor(game_history.done_history).cuda()
    #     # print(len(r))
    #     # print(r)
    #     # print(v_t)
    #     # print('----------------')


    #     prob=self.model(s)[0]
    #     old_log_prob=torch.log(prob.gather(1,a)+1e6)
    #     for _ in range(3):
    #         prob,value=self.model(s)
    #         td_error=v_t-value
    #         # print(prob)
    #         log_prob=torch.log(prob+1e6)
    #         entropy=-(log_prob*prob).sum(1, keepdim=True)
    #         log_prob=log_prob.gather(1,a)

    #         value=value.squeeze()

    #         ratios=torch.exp(log_prob-old_log_prob.detach())
    #         advantage=td_error.detach()
    #         surr1=ratios*advantage
    #         surr2=torch.clamp(ratios,1-self.epsilon,1+self.epsilon)*advantage

    #         # loss=-torch.min(surr1,surr2)+0.5*self.MseLoss(value,v_t)-0.01*entropy
    #         loss=(-torch.min(surr1,surr2)).mean()+(0.5*td_error.pow(2)).mean()-0.01*entropy.mean()

    #         self.opt.zero_grad()
    #         loss.backward()
    #         # print(loss.mean())
    #         self.opt.step()

    #     self.share_storage.set_info.remote({"weights": copy.deepcopy(self.model.get_weights())})


        # probs,value=self.model(s)
        # # print(a)
        # log_prob=torch.log(probs.gather(1,a.unsqueeze(0)))

        # adv=0
        # adv_list=[]
        # td_error=v_t-value
        # for idx in reversed(range(len(r))):
        #     adv=self.gamma*0.95*adv+td_error[idx][0].item()
        #     adv_list.append(adv)
        # adv_list.reverse()
        # advs=torch.FloatTensor(adv_list).cuda()
        # advs = (advs - advs.mean())/(advs.std()+1e-3)
        # c_loss=td_error.pow(2)/2.0
        # a_loss=log_prob*advs.detach()
        # # print(a_loss)

        # self.opt.zero_grad()
        # (a_loss+c_loss).mean().backward()
        # torch.nn.utils.clip_grad_norm_(self.model.parameters(),40)
        # self.opt.step()
        # self.share_storage.set_info.remote({"weights": copy.deepcopy(self.model.get_weights())})
