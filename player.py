from os import PRIO_USER
import ray
import torch
from models import Model
import numpy as np
from atari_wrappers import make_atari, wrap_deepmind,LazyFrames
from torch.distributions import Categorical
import copy

@ray.remote
class Player:
    def __init__(self,checkpoint,output,share_storage,replay_buffer,test_mode,seed):
        torch.manual_seed(seed)
        self.game=make_atari(checkpoint["game"]+"NoFrameskip-v4")
        self.game=wrap_deepmind(self.game, scale = True, frame_stack=True )
        self.game.seed(seed)
        self.action_list=checkpoint["action_list"]
        self.max_len_episode=checkpoint["max_len_episode"]
        self.max_len_step=checkpoint["max_len_step"]
        self.memory_update_iter=checkpoint["memory_update_iter"]
        self.gamma=checkpoint['gamma']
        self.share_storage=share_storage
        self.replay_buffer=replay_buffer
        self.training_step=checkpoint["training_step"]
        self.model=Model().cuda()
        self.model.set_weights(checkpoint["weights"])
        self.output=output
        self.palyed_game=0
        self.game_history=GameHistory(self.memory_update_iter)

        self.test_mode = test_mode
        if self.test_mode:
            self.palyed_game = 0
            self.epr_writer = open('./log/'+checkpoint["game"]+'.log', 'w')
        print('player init done')

        self.obs=self.game.reset()
        self.len_step=0
        self.ep_r=0
        self.played_game=0
        self.done=False

    def continous_self_play(self):

        # print('start play')
        while self.share_storage.get_info.remote("terminate"):
            with torch.no_grad():
                self.model.set_weights(ray.get(self.share_storage.get_info.remote('weights')))

                for step in range(self.memory_update_iter):
                    fake_done=0
                    self.len_step+=1
                    action_index,log_prob,value=self.choose_action(self.obs)
                    obs_,reward,done,info=self.game.step(self.action_list[action_index])
                    
                    '''custom loss'''
                    if reward==-1:
                        fake_done=1
                    ''''''

                    done = done or (self.len_step>=self.max_len_step)
                    self.game_history.store_memory(step,self.process_input(self.obs).squeeze(),value,action_index,reward,log_prob,done)

                    self.obs=obs_
                    self.ep_r+=reward

                    if done:
                        self.obs=self.game.reset()
                        print(self.played_game,self.ep_r,'FPS:',self.len_step)
                        # if self.test_mode:
                        #     self.write_log(self.ep_r)
                        self.len_step=0
                        self.ep_r=0
                        self.played_game+=1
                        # break

                self.game_history.store_obs_(self.process_input(self.obs).cpu())
                self.replay_buffer.upload_memory.remote(copy.deepcopy(self.game_history))

        # if self.test_mode:
        #     self.epr_writer.close()
        # print('end play')

    def write_log(self,ep_r):
        self.epr_writer.write(str(ep_r)+'\n')
        self.epr_writer.flush()


    def choose_action(self,obs):
        prob,value=self.model(self.process_input(obs))
        action_index=prob.multinomial(1)
        log_prob=torch.log(prob.gather(1,action_index))
        return action_index,log_prob,value


    def process_input(self,obs):
        return torch.FloatTensor(obs).cuda().permute(2,0,1).unsqueeze(0)



class GameHistory:
    def __init__(self,memory_update_iter) -> None:
        self.obs_history=torch.zeros(memory_update_iter+1,4,84,84)
        self.value_history=torch.zeros(memory_update_iter,1)
        self.action_history=torch.zeros(memory_update_iter,1)
        self.reward_history=torch.zeros(memory_update_iter,1)
        self.log_prob_history=torch.zeros(memory_update_iter,1)
        self.done_history=torch.zeros(memory_update_iter,1)


    def store_memory(self,step,obs,value,action,reward,log_prob,done):
        self.obs_history[step]=obs
        self.value_history[step]=value
        self.action_history[step]=action
        self.reward_history[step]=reward
        self.log_prob_history[step]=log_prob
        self.done_history[step]=done
        

    def store_obs_(self,obs_):
        self.obs_history[-1]=obs_


    def cuda(self):
        self.obs_history=self.obs_history.cuda()
        self.value_history=self.value_history.cuda()
        self.action_history=self.action_history.cuda()
        self.reward_history=self.reward_history.cuda()
        self.log_prob_history=self.log_prob_history.cuda()
        self.done_history=self.done_history.cuda()
