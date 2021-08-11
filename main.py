import os
from server import Server
import ray
import numpy as np
import torch
import player
import trainer
import replay_buffer
import share_storage
from models import Model
import copy
import time

NUM_WORKER=8
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
        "weights":None,
        "lr":2.5e-4,
        "terminate":False,
        "start_training":False,
        "training_step":0,
        "max_training_step":2e3,
        "memory_update_iter":128,
        "max_len_episode":2e3,
        "memory_size":1000,
        "batch_size":256,
        "epsilon":0.1,
        "gamma":0.99,
        "entropy_beta":1e-3,
        "model_save_iter":1,
        "tau":0.002,
        "max_len_step":3e3,
        "action_list":[0,1,2,3,4,5]
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

    replay_buffer_worker = replay_buffer.ReplayBuffer.remote(checkpoint, share_storage_worker)
    
    training_worker=trainer.Trainer.options(num_cpus=1,num_gpus=0.1).remote(checkpoint,share_storage_worker,replay_buffer_worker)

    self_play_workers = [player.Player.options(num_cpus=1,num_gpus=0.1).remote(checkpoint,True,share_storage_worker,replay_buffer_worker,False,seed) 
                                        for seed in range(NUM_WORKER)]

    server_worker = Server.remote(checkpoint,self_play_workers,training_worker)
    print("init nodes done!")
    

    # start works
    server_worker.run.remote()
    # [worker.continous_self_play.remote() for worker in self_play_workers]
    # training_worker.continous_update_weights.remote()

    while True:
        time.sleep(5)

    print('Done!')

