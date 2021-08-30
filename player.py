import time
import ray
import torch
from models import Model
import numpy as np
from atari_wrappers import make_atari, wrap_deepmind,LazyFrames
from torch.distributions import Categorical
import copy

@ray.remote
class Player:
    def __init__(self,checkpoint,share_storage,replay_buffer,test_mode,seed):
        torch.manual_seed(seed)
        self.game=make_atari(checkpoint["game"]+"NoFrameskip-v4")
        self.game=wrap_deepmind(self.game, scale = True, frame_stack=True )
        self.game.seed(seed)
        self.seed=seed
        self.action_list=checkpoint["action_list"]
        self.max_len_episode=checkpoint["max_len_episode"]
        self.max_len_step=checkpoint["max_len_step"]
        self.len_episode=checkpoint["len_episode"]
        self.gamma=checkpoint['gamma']
        self.share_storage=share_storage
        self.replay_buffer=replay_buffer
        self.training_step=checkpoint["training_step"]
        self.model=Model().cuda()
        self.model.set_weights(checkpoint["weights"])
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
        while self.share_storage.get_info.remote("terminate"):
            with torch.no_grad():
                self.model.set_weights(ray.get(self.share_storage.get_info.remote('weights')))
                a=time.time()
                for step in range(self.len_episode):
                    fake_done=0
                    self.len_step+=1
                    action_index,log_prob=self.choose_action(self.obs)
                    obs_,reward,done,info=self.game.step(self.action_list[action_index])

                    '''custom loss'''
                    # if info["ale.lives"] != self.live:
                    #     reward=-1
                    #     self.live=info["ale.lives"]
                    #     self.game.step(1)
                    #     fake_done=1
                    ''''''

                    done = done or (self.len_step>=self.max_len_step)
                    self.game_history.store_memory(step,self.obs.squeeze(),action_index,reward,log_prob,done)

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
                        # break
                b=time.time()
                # print('------------------------------------------------------------------------',b-a)
                self.game_history.store_obs_(self.obs)
                self.replay_buffer[0].upload_memory.remote(copy.deepcopy(self.game_history))

    def write_log(self,ep_r):
        self.epr_writer.write(str(ep_r)+'\n')
        self.epr_writer.flush()


    def choose_action(self,obs):
        prob,_=self.model(obs.cuda().unsqueeze(0))
        action_index=prob.multinomial(1)
        log_prob=torch.log(prob.gather(1,action_index))
        return action_index.cpu(),log_prob.cpu()


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
