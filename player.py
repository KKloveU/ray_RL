import ray
import torch
from models import Actor, Critic
import numpy as np
from atari_wrappers import make_atari, wrap_deepmind,LazyFrames
from torch.distributions import Categorical


@ray.remote
class Player:
    def __init__(self,checkpoint,output,share_storage,replay_buffer,test_mode):
        self.game=make_atari(checkpoint["game"]+"NoFrameskip-v4")
        self.game=wrap_deepmind(self.game, scale = True, frame_stack=True )
        self.action_list=checkpoint["action_list"]
        self.max_len_episode=checkpoint["max_len_episode"]
        self.max_len_step=checkpoint["max_len_step"]
        self.memory_update_iter=checkpoint["memory_update_iter"]
        self.gamma=checkpoint['gamma']
        self.share_storage=share_storage
        self.replay_buffer=replay_buffer
        self.training_step=checkpoint["training_step"]
        self.actor=Actor().cuda()
        self.actor.set_weights(checkpoint["actor_weights"])
        self.output=output
        self.palyed_game=0
        self.game_history=GameHistory()

        self.test_mode = test_mode
        if self.test_mode:
            self.palyed_game = 0
            self.epr_writer = open('./log/'+checkpoint["game"]+'.log', 'w')
        print('player init done')

        
    def continous_self_play(self):

        print('start play')
        while not ray.get(self.share_storage.get_info.remote("terminate")):
            self.actor.set_weights(ray.get(self.share_storage.get_info.remote('actor_weights')))
            self.paly_game()

        if self.test_mode:
            self.epr_writer.close()
        print('end play')

    def paly_game(self):
        ep_r=0
        done=False
        step=1
        obs=self.game.reset()
        self.game_history.clear_memory()
        
        while not done :
            fake_done=False
            action_index=self.choose_action(obs)
            obs_,reward,done,info=self.game.step(self.action_list[action_index])
            if reward==-1:
                fake_done=True

            self.game_history.save_transition(obs,action_index.data,reward,obs_,fake_done)
            
            if fake_done or step % self.memory_update_iter == 0:
                self.replay_buffer.store_memory.remote(self.game_history)
                self.game_history.clear_memory()
                
            obs=obs_
            ep_r+=reward
            step+=1

        if self.output:
            print(self.palyed_game,ep_r)
        
        if self.test_mode:
            print(self.palyed_game, ep_r)
            self.epr_writer.write(str(ep_r)+'\n')
            self.epr_writer.flush()
            self.palyed_game += 1
        
        self.palyed_game+=1

    def choose_action(self,obs):
        obs_input=torch.FloatTensor(obs).cuda().permute(2,0,1).unsqueeze(0)
        prob=self.actor(obs_input)
        action_index=prob.multinomial(1)
        # m=Categorical(prob)
        # action_index=m.sample()
        return action_index

class GameHistory:
    def __init__(self) -> None:
        # self.trans_history=[]
        self.obs_history=[]
        self.act_history=[]
        self.reward_history=[]
        self.vtarget_history=[]
        self.done_history=[]
        self.obs_next_history=[]

    def save_transition(self,obs,action,reward,obs_,done):
        action=action.item()
        self.obs_history.append(obs)
        self.act_history.append(action)
        self.reward_history.append(reward)
        self.obs_next_history.append(obs_)
        self.done_history.append(done)
        
    def clear_memory(self):
        self.obs_history=[]
        self.act_history=[]
        self.reward_history=[]
        self.vtarget_history=[]
        self.done_history=[]
        self.obs_next_history=[]

    def process(self,v_s_,gamma):
        self.vtarget_history=[]
        for r in self.reward_history[::-1]:
            v_s_=r+gamma*v_s_
            self.vtarget_history.append(v_s_)
        self.vtarget_history.reverse()

        # self.action_history=torch.LongTensor(self.action_history).cuda()
        # self.obs_history=torch.stack(self.obs_history).cuda()
        # self.vtarget_history=torch.FloatTensor(self.vtarget_history).cuda()
