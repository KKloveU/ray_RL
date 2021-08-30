from os import PRIO_USER
import time
import ray
import torch
from models import Model
import numpy as np
from atari_wrappers import make_atari, wrap_deepmind,LazyFrames
from torch.distributions import Categorical
import copy


@ray.remote
class Rollout:
    def __init__(self,checkpoint,share_storage,replay_buffer,test_mode) -> None:
        self.share_storage=share_storage
        self.replay_buffer=replay_buffer
        self.num_workers=checkpoint['num_workers']
        self.model=Model().cuda()
        self.model.set_weights(checkpoint["weights"])
        
        self.obs_history=[]
        self.action_history=[]
        self.reward_history=[]
        self.log_prob_history=[]
        self.done_history=[]
        print('rollout init done')

    def run(self,play_workers):
        # print('run!')
        self.play_workers=play_workers
        # self.a=time.time()
        [worker.continous_self_play.remote() for worker in self.play_workers]
    
    def choose_action(self,obs):
        with torch.no_grad():
            prob,_=self.model(obs.cuda())
            action_index=prob.multinomial(1)
            log_prob=torch.log(prob.gather(1,action_index))
            # print(action_index,log_prob)
            return action_index.cpu(),log_prob.cpu()
    
    def merge_memory(self,traj):
        self.obs_history.append(traj.obs_history)
        self.action_history.append(traj.action_history)
        self.reward_history.append(traj.reward_history)
        self.log_prob_history.append(traj.log_prob_history)
        self.done_history.append(traj.done_history)

        if len(self.obs_history)==self.num_workers:
            obs_memory=torch.stack(self.obs_history)
            act_memory=torch.stack(self.action_history)
            reward_memory=torch.stack(self.reward_history)
            log_prob_memory=torch.stack(self.log_prob_history)
            done_memory=torch.stack(self.done_history)

            self.replay_buffer.upload_memory.remote((obs_memory,act_memory,reward_memory,log_prob_memory,done_memory))
            self.model.set_weights(ray.get(self.share_storage.get_info.remote('weights')))

            self.obs_history=[]
            self.action_history=[]
            self.reward_history=[]
            self.log_prob_history=[]
            self.done_history=[]
            # self.b=time.time()
            # print(self.b-self.a)
            # self.a=self.b
            [worker.continous_self_play.remote() for worker in self.play_workers]
            # print('upload memory!')



@ray.remote
class Player:
    def __init__(self,checkpoint,server,test_mode,seed):
        self.game=make_atari(checkpoint["game"]+"NoFrameskip-v4")
        self.game=wrap_deepmind(self.game, scale = True, frame_stack=True )
        self.game.seed(seed)
        self.seed=seed
        self.action_list=checkpoint["action_list"]
        self.max_len_episode=checkpoint["max_len_episode"]
        self.max_len_step=checkpoint["max_len_step"]
        self.len_episode=checkpoint["len_episode"]
        self.gamma=checkpoint['gamma']
        self.training_step=checkpoint["training_step"]
        self.server=server
        self.palyed_game=0
        self.game_history=GameHistory(self.len_episode)

        self.test_mode = test_mode
        if self.test_mode:
            self.epr_writer = open('./log/'+checkpoint["game"]+str(seed)+'.log', 'w')
        print('player init done')

        self.obs=self.game.reset()
        # self.game.step(1)
        self.len_step=0
        self.ep_r=0
        self.played_game=0
        self.done=False
        self.live=5

    def continous_self_play(self):
        # print('start play')
        self.a=time.time()
        for step in range(self.len_episode):
            fake_done=0
            self.len_step+=1
            
            action_index,log_prob=ray.get(self.server.choose_action.remote(self.obs.unsqueeze(0)))
            obs_,reward,done,info=self.game.step(self.action_list[action_index])
            
            '''custom loss'''
            # if info["ale.lives"] != self.live:
            #     reward=-1
            #     self.live=info["ale.lives"]
            #     self.game.step(1)
            #     fake_done=1
            ''''''
            
            done = done or (self.len_step>=self.max_len_step)
            self.game_history.store_memory(step,self.obs,action_index,reward,log_prob,done)

            self.obs=obs_
            self.ep_r+=reward
            

            if done:
                self.obs=self.game.reset()
                # self.game.step(1)
                print(self.played_game,self.ep_r,'FPS:',self.len_step)
                if self.test_mode:
                    self.write_log(self.ep_r)
                self.len_step=0
                self.ep_r=0
                self.played_game+=1
                self.live=5
        self.b=time.time()
        self.game_history.store_obs_(self.obs)
        self.server.merge_memory.remote(self.game_history)
        
        # print(self.b-self.a)


    def write_log(self,ep_r):
        self.epr_writer.write(str(ep_r)+'\n')
        self.epr_writer.flush()



class GameHistory:
    def __init__(self,len_episode) -> None:
        self.obs_history=torch.zeros(len_episode+1,4,84,84)
        self.action_history=torch.zeros(len_episode,1)
        self.reward_history=torch.zeros(len_episode,1)
        self.log_prob_history=torch.zeros(len_episode,1)
        self.done_history=torch.zeros(len_episode,1)


    def store_memory(self,step,obs,action,reward,log_prob,done):
        self.obs_history[step]=obs
        self.action_history[step]=action
        self.reward_history[step]=reward
        self.log_prob_history[step]=log_prob
        self.done_history[step]=done
        

    def store_obs_(self,obs_):
        self.obs_history[-1]=obs_


    # def cuda(self):
    #     self.obs_history=self.obs_history.cuda()
    #     self.action_history=self.action_history.cuda()
    #     self.reward_history=self.reward_history.cuda()
    #     self.log_prob_history=self.log_prob_history.cuda()
    #     self.done_history=self.done_history.cuda()