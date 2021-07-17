from numpy.core.shape_base import vstack
from numpy.lib.utils import lookfor
import ray
import torch
from models import Model
import numpy as np
from atari_wrappers import make_atari, wrap_deepmind,LazyFrames


@ray.remote
class Player:
    def __init__(self,checkpoint,output,share_storage,trainer):
        self.game=make_atari(checkpoint["game"]+"NoFrameskip-v4")
        self.game=wrap_deepmind(self.game, scale = True, frame_stack=True )
        self.action_list=checkpoint["action_list"]
        self.max_len_episode=checkpoint["max_len_episode"]
        self.max_len_step=checkpoint["max_len_step"]
        self.gamma=checkpoint['gamma']
        self.share_storage=share_storage
        self.trainer=trainer
        self.training_step=checkpoint["training_step"]
        self.model=Model().cuda()
        self.model.set_weights(checkpoint["weights"])
        self.output=output
        self.palyed_game=0
        self.game_history=GameHistory()
        print('player init done')
        
    def continous_self_play(self,test_mode=False):
        print('start play')
        while not ray.get(self.share_storage.get_info.remote("terminate")):
            if test_mode:   # Take the best action
                self.paly_game(False)
            else:   # collect training data
                self.paly_game(False)
        print('end play')

    def paly_game(self,render):
        
        ep_r=0
        done=False
        obs=self.game.reset()
        # self.game.step(1)
        step=1
        # live=5
        self.game_history.clear_memory()
        
        while not done :
            # fake_done=False
            action_index=self.choose_action(obs)
            obs_,reward,done,info=self.game.step(self.action_list[action_index])
            # if info["ale.lives"] != live:
            #     reward=-10
            #     live=info["ale.lives"]
            #     self.game.step(1)
            #     fake_done=True
            self.game_history.save_transition(torch.FloatTensor(obs).cuda().permute(2,0,1),action_index,reward)
            
            if done or step%20==0:
                if done:
                    v_s_=0.
                else:
                    _,v_s_=self.model(torch.FloatTensor(obs_).cuda().permute(2,0,1).unsqueeze(0))
                    v_s_=v_s_[0]
                    
                self.game_history.process(v_s_,self.gamma)
                self.trainer.update_weights.remote(self.game_history)
                self.model.set_weights(ray.get(self.share_storage.get_info.remote('weights')))
                self.game_history.clear_memory()
                
            obs=obs_
            ep_r+=reward
            step+=1

        if self.output:
            print(self.palyed_game,ep_r)
        self.palyed_game+=1

    def choose_action(self,obs):
        obs_input=torch.FloatTensor(obs).cuda().permute(2,0,1).unsqueeze(0)
        prob,_=self.model(obs_input)
        action_index=prob[0].multinomial(num_samples=1).detach()
        return action_index


class GameHistory:
    def __init__(self) -> None:
        # self.trans_history=[]
        self.obs_history=[]
        self.action_history=[]
        self.reward_history=[]
        self.v_target_history=[]

    def save_transition(self,obs,action,reward):
        self.obs_history.append(obs)
        self.action_history.append(action)
        self.reward_history.append(reward)
        
    def clear_memory(self):
        self.obs_history=[]
        self.action_history=[]
        self.reward_history=[]
        self.v_target_history=[]

    def process(self,v_s_,gamma):

        self.v_target_history=[]
        for r in self.reward_history[::-1]:
            v_s_=r+gamma*v_s_
            self.v_target_history.append(v_s_)
        self.v_target_history.reverse()

        self.action_history=torch.LongTensor(self.action_history).cuda()
        self.obs_history=torch.stack(self.obs_history).cuda()
        self.v_target_history=torch.FloatTensor(self.v_target_history).cuda()