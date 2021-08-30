import os

from ray import worker
from rollout import Player, Rollout
from server import Server
import player
import ray
import numpy as np
import torch
# import player
import trainer
import replay_buffer
import share_storage
from models import Model
import copy
import time
import rollout_x
game_name="Pong"
path_file=r"./model_checkpoint/"+game_name+"_model" 

class ModelLoader:
    # Trick to force DataParallel to stay on CPU to get weights on CPU even if there is a GPU
    def __init__(self):
        pass

    def get_initial_weights(self):
        model=Model()
        weigths = model.get_weights()
        # summary = str(model).replace("\n", " \n\n")
        return weigths

if __name__=="__main__":
    ray.init()
    checkpoint={
        "game":game_name,
        "num_workers":8,
        "num_cluster":2,
        "num_reuse":3,
        "weights":None,
        "lr":2.5e-4,
        "terminate":False,
        "start_training":False,
        "training_step":0,
        "max_training_step":2e3,
        "len_episode":128,
        "max_len_episode":2e3,
        "memory_size":10, 
        "batch_size":256,
        "epsilon":0.2,
        "gamma":0.99,
        "gae_lambda":0.95,
        "entropy_coef":1e-2,
        "value_loss_coef":0.5,
        "model_save_iter":1,
        "tau":0.002,
        "max_len_step":2e4,
        "action_list":[0,2,3]
    }

    # if os.path.exists(path_file):
    #     model_params=torch.load(path_file) 
    #     checkpoint["actor_weights"]=copy.deepcopy(model_params["actor_weights"])
    #     checkpoint["critic_weights"]=copy.deepcopy(model_params["critic_weights"])
    #     print('Load model!')

    # else:
    #     cpu_actor=ModelLoader()
    #     actor_weights,critic_weights = cpu_actor.get_initial_weights()
    #     checkpoint["actor_weights"] = copy.deepcopy(actor_weights)
    #     checkpoint["critic_weights"]=copy.deepcopy(critic_weights)
    #     print('Init model!')

    cpu_actor=ModelLoader()
    weights = cpu_actor.get_initial_weights()
    checkpoint["weights"] = copy.deepcopy(weights)
    print('Init model!')
 
    # init nodes
    share_storage_worker = share_storage.SharedStorage.remote(checkpoint)

    replay_buffer_worker = [replay_buffer.ReplayBuffer.remote(checkpoint, share_storage_worker) for _ in range(2)]
    
    
    training_worker=trainer.Trainer.options(num_cpus=1,num_gpus=0.1).remote(checkpoint,share_storage_worker,replay_buffer_worker)


    '''rollout block'''
    # rollout1=rollout_x.Rollout.options(num_cpus=1).remote(checkpoint,share_storage_worker,replay_buffer_worker[0])
    # self_play_workers1 = [rollout_x.Player.options(num_cpus=1,num_gpus=0.1).remote(checkpoint,rollout1,True,seed) 
    #                                     for seed in range(checkpoint["num_workers"]*checkpoint["num_cluster"])] 

    # rollout2=rollout_x.Rollout.options(num_cpus=1).remote(checkpoint,share_storage_worker,replay_buffer_worker[1])
    # self_play_workers2 = [rollout_x.Player.options(num_cpus=1,num_gpus=0.1).remote(checkpoint,rollout2,False,seed) 
    #                                     for seed in range(checkpoint["num_workers"]*checkpoint["num_cluster"])]   
    print("init nodes done!")
    
    self_play_workers=[player.Player.options(num_cpus=1,num_gpus=0.1).remote(checkpoint,share_storage_worker,replay_buffer_worker,False,seed) for seed in range(checkpoint["num_workers"]*checkpoint["num_cluster"])]  

    # start works
    # rollout1.run.remote(self_play_workers1)
    # rollout2.run.remote(self_play_workers2)
    [worker.continous_self_play.remote() for worker in self_play_workers]
    training_worker.continous_update_weights.remote()

    while True:
        time.sleep(5)

    print('Done!')

