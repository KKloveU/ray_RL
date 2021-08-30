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
    def __init__(self,checkpoint,share_storage,replay_buffer) -> None:
        self.share_storage=share_storage
        self.replay_buffer=replay_buffer
        self.num_workers=checkpoint['num_workers']
        self.num_cluster=checkpoint['num_cluster']
        self.weights=checkpoint['weights']

        self.worker_list=[[] for _ in range(self.num_cluster)]
        self.obs_history=[[] for _ in range(self.num_cluster)]
        self.action_history=[[] for _ in range(self.num_cluster)]
        self.reward_history=[[] for _ in range(self.num_cluster)]
        self.log_prob_history=[[] for _ in range(self.num_cluster)]
        self.done_history=[[] for _ in range(self.num_cluster)]
        print('rollout init done')
        self.test=0
    def run(self,play_workers):
        i=0
        for j in range(self.num_cluster):
            for _ in range(self.num_workers):
                self.worker_list[j].append(play_workers[i])
                i+=1

        [worker.continous_self_play.remote(self.weights) for worker in play_workers]
        print(self.worker_list)


    def merge_memory(self,rank,traj):
        group=rank//self.num_workers
        self.obs_history[group].append(traj.obs_history)
        self.action_history[group].append(traj.action_history)
        self.reward_history[group].append(traj.reward_history)
        self.log_prob_history[group].append(traj.log_prob_history)
        self.done_history[group].append(traj.done_history)

        if len(self.obs_history[group])==self.num_workers:

            self.weights=ray.get(self.share_storage.get_info.remote('weights'))
            [worker.continous_self_play.remote(self.weights) for worker in self.worker_list[group]]

            obs_memory=torch.stack(self.obs_history[group])
            act_memory=torch.stack(self.action_history[group])
            reward_memory=torch.stack(self.reward_history[group])
            log_prob_memory=torch.stack(self.log_prob_history[group])
            done_memory=torch.stack(self.done_history[group])

            self.replay_buffer.upload_memory.remote((obs_memory,act_memory,reward_memory,log_prob_memory,done_memory))

            self.obs_history[group]=[]
            self.action_history[group]=[]
            self.reward_history[group]=[]
            self.log_prob_history[group]=[]
            self.done_history[group]=[]


@ray.remote
class Player:
    def __init__(self,checkpoint,server,test_mode,seed):
        # torch.manual_seed(seed)
        self.game=make_atari(checkpoint["game"]+"NoFrameskip-v4")
        self.game=wrap_deepmind(self.game, scale = True, frame_stack=True )
        # self.game.seed(seed)
        self.rank=seed
        self.action_list=checkpoint["action_list"]
        self.max_len_episode=checkpoint["max_len_episode"]
        self.max_len_step=checkpoint["max_len_step"]
        self.len_episode=checkpoint["len_episode"]
        self.training_step=checkpoint["training_step"]
        self.gamma=checkpoint['gamma']
        self.server=server
        self.model=Model().cuda()
        self.palyed_game=0
        self.game_history=GameHistory(self.len_episode)

        self.test_mode = test_mode
        if self.test_mode:
            self.epr_writer = open('./log/'+checkpoint["game"]+str(seed)+'.log', 'w')
        print('player init done')

        self.obs=self.game.reset()
        self.len_step=0
        self.ep_r=0
        self.played_game=0
        self.done=False
        self.live=5

    def continous_self_play(self,weights):
        # print('start play')
        with torch.no_grad():
            self.model.set_weights(weights)
            a=time.time()
            for step in range(self.len_episode):
                fake_done=0
                self.len_step+=1
                action_index,log_prob,_=self.choose_action(self.obs)
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
                    print(self.played_game,self.ep_r,'FPS:',self.len_step)
                    if self.test_mode:
                        self.write_log(self.ep_r)
                    self.len_step=0
                    self.ep_r=0
                    self.played_game+=1
                    self.live=5
                b=time.time()
                # print('------------------------------------------------------------------------',b-a)
            self.game_history.store_obs_(self.obs)
            self.server.merge_memory.remote(self.rank,self.game_history)

    def write_log(self,ep_r):
        self.epr_writer.write(str(ep_r)+'\n')
        self.epr_writer.flush()


    def choose_action(self,obs):
        prob,value=self.model(obs.cuda().unsqueeze(0))
        action_index=prob.multinomial(1)
        log_prob=torch.log(prob.gather(1,action_index))
        return action_index.cpu(),log_prob.cpu(),value.cpu()


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


    def cuda(self):
        self.obs_history=self.obs_history.cuda()
        self.action_history=self.action_history.cuda()
        self.reward_history=self.reward_history.cuda()
        self.log_prob_history=self.log_prob_history.cuda()
        self.done_history=self.done_history.cuda()
