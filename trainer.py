import time
import ray
import cv2
import models
import copy
import torch
import numpy as np

@ray.remote
class Trainer:
    def __init__(self,checkpoint) -> None:
        self.eval_model=models.Model()
        self.target_model=models.Model()
        self.eval_model.set_weights(copy.deepcopy(checkpoint["weights"]))
        self.target_model.set_weights(copy.deepcopy(checkpoint["weights"]))
        self.eval_model.cuda()
        self.target_model.cuda()
        
        # self.eval_model.train()
        # self.target_model.eval()

        self.gamma=checkpoint['gamma']
        self.training_step=checkpoint['training_step']
        self.trained_step=checkpoint['max_training_step']
        self.replace_target_iter=checkpoint['replace_target_iter']
        self.tau=checkpoint["tau"]
        self.batch_size=checkpoint['batch_size']
        self.learn_step_counter=1
        self.loss_fn=torch.nn.SmoothL1Loss(reduction="none")

        self.optimizer=torch.optim.Adam(self.eval_model.parameters(),lr=checkpoint['lr'])
        print('trainer init done')
    
    def continous_update_weights(self,share_storage,repaly_buffer):
        print('wait train')
        while not ray.get(share_storage.get_info.remote('start_training')):
            time.sleep(0.1)
        print('start train-----------------------------------------------------')
        
        while True:

            tree_idx, batch_memory, ISWeights=ray.get(repaly_buffer.get_batch.remote(self.batch_size))

            self.learn_step_counter=self.learn_step_counter%self.replace_target_iter
            if self.learn_step_counter==0:
                share_storage.set_info.remote({"weights": copy.deepcopy(self.eval_model.get_weights())})
                share_storage.save_checkpoint.remote()
                print('net_replace!!!!') 
                
            tree_idx,abs_error=self.update_weights(tree_idx, batch_memory, ISWeights)
            repaly_buffer.batch_update.remote(tree_idx,abs_error)
            for target_param, param in zip(self.target_model.parameters(), self.eval_model.parameters()):
                target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)
            self.learn_step_counter+=1

    def update_weights(self,tree_idx, batch_memory, ISWeights):
        batch=copy.deepcopy(batch_memory)
        weight=copy.deepcopy(ISWeights)
        batch_obs=[]
        batch_act=[]
        batch_reward=[]
        batch_obs_=[]
        for i in range(self.batch_size):
            batch_obs.append(batch[i,0][0])
            batch_act.append(batch[i,0][1])
            batch_reward.append(batch[i,0][2])
            batch_obs_.append(batch[i,0][3])
        
        batch_obs=torch.FloatTensor(np.stack(batch_obs)).permute(0,3,1,2).cuda()
        batch_act=torch.LongTensor(np.vstack(batch_act)).cuda()
        batch_reward=torch.FloatTensor(np.vstack(batch_reward)).cuda()
        batch_obs_=torch.FloatTensor(np.stack(batch_obs_)).permute(0,3,1,2).cuda()
        batch_weight=torch.FloatTensor(weight).cuda()

        q_eval=self.eval_model(batch_obs).gather(1,batch_act)
        q_next=self.target_model(batch_obs_)
        q_target=batch_reward+self.gamma*q_next.max(1)[0].view(-1,1)

        # batch_weight.mean()
        loss = (batch_weight * self.loss_fn(q_eval,q_target.detach())).mean()

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.eval_model.parameters():                 ######
            param.grad.data.clamp_(-1, 1) 
        self.optimizer.step()

        abs_error=torch.abs(q_eval-q_target).detach().cpu().numpy()
        return tree_idx,abs_error


        

        