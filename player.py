import ray
import torch
from models import Model
import numpy as np
from atari_wrappers import make_atari, wrap_deepmind,LazyFrames
from torch.distributions import Categorical


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
        self.game_history=GameHistory()

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

    def continous_self_play(self,weights):

        # print('start play')

        with torch.no_grad():
            self.model.set_weights(weights)
            # self.model.set_weights(ray.get(self.share_storage.get_info.remote('weights')))

            
            # while not ray.get(self.share_storage.get_info.remote("terminate")):
            # self.model.set_weights(ray.get(self.share_storage.get_info.remote('weights')))
            self.game_history.clear_memory()

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

                self.game_history.store_memory(self.process_input(self.obs).squeeze(),value,action_index,reward,log_prob,done)

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

                # if fake_done==1:
                #     break


            _,next_value=self.model(self.process_input(self.obs))
            next_value=next_value.item()
            self.game_history.process(next_value,self.gamma)

            self.replay_buffer.upload_memory.remote(self.game_history)

                
        # if self.test_mode:
        #     self.epr_writer.close()
        # print('end play')


    def write_log(self,ep_r):
        self.epr_writer.write(str(ep_r)+'\n')
        self.epr_writer.flush()


    def choose_action(self,obs):
        prob,value=self.model(self.process_input(obs))
        # prob=torch.clamp(prob,1e-3,1-1e-3)
        # print(prob)
        action_index=prob.multinomial(1)
        log_prob=torch.log(prob.gather(1,action_index))
        return action_index,log_prob,value


    def process_input(self,obs):
        return torch.FloatTensor(obs).cuda().permute(2,0,1).unsqueeze(0)

class GameHistory:
    def __init__(self) -> None:
        # self.trans_history=[]
        self.obs_history=[]
        self.value_history=[]
        self.action_history=[]
        self.reward_history=[]
        self.log_prob_history=[]
        self.done_history=[]
        self.returns_history=[]
        self.advantage_history=[]

    def store_memory(self,obs,value,action,reward,log_prob,done):
        self.obs_history.append(obs.cpu().numpy())
        self.value_history.append(value.item())
        self.action_history.append(action.item())
        self.reward_history.append(reward)
        self.log_prob_history.append(log_prob.item())
        self.done_history.append(done)

    def clear_memory(self):
        self.obs_history=[]
        self.value_history=[]
        self.action_history=[]
        self.reward_history=[]
        self.log_prob_history=[]
        self.done_history=[]
        self.returns_history=[]
        self.advantage_history=[]

    def process(self,v_s_,gamma):
        gae=0
        gae_lambda=0.95
        for r,v,done in zip(self.reward_history[::-1],self.value_history[::-1],self.done_history[::-1]):
            delta=r+v_s_*gamma*(1-done)-v
            gae=delta+gamma*gae_lambda*(1-done)*gae
            self.returns_history.append(gae+v)
            v_s_=v
        self.returns_history.reverse()

        for i in range(len(self.reward_history)):
            self.advantage_history.append(self.returns_history[i]-self.value_history[i])
        
        self.advantage_history=torch.FloatTensor(self.advantage_history)
        self.advantage_history=(self.advantage_history-self.advantage_history.mean())/(self.advantage_history.std()+1e-5)
        self.advantage_history=list(self.advantage_history.numpy())

        # print(self.done_history)
        # print(self.returns_history)
        # print('----------------')

        # self.vtarget_history = torch.FloatTensor(self.vtarget_history)
        # self.vtarget_history = (self.vtarget_history-self.vtarget_history.mean())/(self.vtarget_history.std()+1e-7)
        # self.vtarget_history = list(self.vtarget_history.numpy())


    # def get_gae(self, states, rewards, next_states, dones):
    #     prob,values = self.model(states)
    #     td_target = rewards + self.gamma * self.model(next_states)[1] * (1 - dones)
    #     td_error = td_target.detach() - values
    #     # delta = delta.detach().cpu().numpy()
    #     advantage_lst = []
    #     advantage = 0.0
    #     for idx in reversed(range(len(td_error))):
    #         if dones[idx] == 1:
    #             advantage = 0.0
    #         advantage = self.gamma * 0.95 * advantage + td_error[idx][0]
    #         advantage_lst.append([advantage])
    #     advantage_lst.reverse()
    #     advantages = torch.FloatTensor(advantage_lst).cuda()
    #     advantages = (advantages-advantages.mean()) / advantages.std()
    #     return prob, td_error, advantages.detach()
