import time
import numpy as np
import torch
from atari_wrappers import make_atari, wrap_deepmind
from models import Model

game_name="Breakout"
path_file=r"./model_checkpoint/"+game_name+"_model"
game=make_atari(game_name+'NoFrameskip-v4')

game=wrap_deepmind(game, scale = True, frame_stack=True)
model=Model().cuda()
print(game.action_space)
model_params=torch.load(path_file) 
model.set_weights(model_params["weights"])
action_list=[0,2,3]
model.eval()

with torch.no_grad():
    for episode in range(10):
        ep_r=0
        done=False
        live=5
        obs=game.reset()
        game.step(1)
        while not done :
            game.render()
            time.sleep(0.01)
            obs_input=torch.FloatTensor(np.array(obs)).cuda().permute(2,0,1).unsqueeze(0)
            action_index=np.argmax(model(obs_input).cpu().numpy()[0])
            obs_,reward,done,info=game.step(action_list[action_index])
            if info["ale.lives"] != live:
                    reward=-1 
                    live=info["ale.lives"]
                    game.step(1)
            print(info)
            obs=obs_
            ep_r+=reward
            
        print(ep_r)
