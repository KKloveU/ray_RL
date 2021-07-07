import ray
import gym
import torch
from models import Model
import numpy as np
import random
from atari_wrappers import make_atari, wrap_deepmind
import time

@ray.remote
class Player:
    def __init__(self,checkpoint,output,episilon_decay):
        self.game=make_atari(checkpoint["game"])
        self.game=wrap_deepmind(self.game, scale = True, frame_stack=True )
        self.action_list=[0,2,3]
        self.epsilon=checkpoint['epsilon']
        self.max_len_episode=checkpoint["max_len_episode"]
        self.checkpoint=checkpoint
        self.training_step=checkpoint["training_step"]
        self.model=Model().cuda()
        self.model.set_weights(checkpoint["weights"])
        self.model.eval()
        self.output=output
        self.palyed_game=0
        self.epsilon_decay=episilon_decay
        print('player init done')


    def continous_self_play(self,share_storage,replay_buffer,test_mode=False):
        print('start play')
        while not ray.get(share_storage.get_info.remote("terminate")):
            self.model.set_weights(ray.get(share_storage.get_info.remote('weights')))

            if test_mode:   # Take the best action
                self.paly_game(True,replay_buffer)
            else:   # collect training data
                self.paly_game(True,replay_buffer)
            if ray.get(share_storage.get_info.remote("start_training"))==True:
                self.epsilon=self.epsilon*(1-self.epsilon_decay) 
        print('end play')

    def paly_game(self,render,replay_buffer):
        game_history=GameHistory()
        ep_r=0
        done=False
        obs=self.game.reset()
        self.game.step(1)
        step=0
        live=5
        with torch.no_grad():
            while not done :
                action_index=self.choose_action(np.array(obs))
                obs_,reward,done,info=self.game.step(self.action_list[action_index])
                if info["ale.lives"] != live:
                    reward=-10
                    live=info["ale.lives"]
                    self.game.step(1)
                game_history.save_transition(np.array(obs),action_index,reward,np.array(obs_))
                if done or step%500==0:
                    replay_buffer.store_memory.remote(game_history)
                    game_history.clear_memory()
                obs=obs_
                ep_r+=reward
                step+=1

        if self.output:
            print(self.palyed_game,ep_r,self.epsilon)
        self.palyed_game+=1

    def choose_action(self,obs):
        if random.random()>self.epsilon:
            obs_input=torch.FloatTensor(obs).permute(2,0,1).unsqueeze(0).cuda()
            action_index=np.argmax(self.model(obs_input).cpu().numpy()[0])
        else:
            action_index=random.randint(0,len(self.action_list)-1)
        return action_index

class GameHistory:
    def __init__(self) -> None:
        self.trans_history=[]
    #     self.obs_history=[]
    #     self.act_history=[]
    #     self.reward_history=[]
    #     self.obs_next_history=[]
    def save_transition(self,obs,a,r,obs_):
        self.trans_history.append([obs,a,r,obs_])
        # self.obs_history.append(obs)
        # self.act_history.append(a)
        # self.reward_history.append(r)
        # self.obs_next_history.append(obs_)
    def clear_memory(self):
        self.trans_history=[]
        # self.obs_history=[]
        # self.act_history=[]
        # self.reward_history=[]
        # self.obs_next_history=[]

    