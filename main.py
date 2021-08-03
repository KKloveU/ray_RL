import os
import ray
import numpy as np
import torch
import player
import trainer
import replay_buffer
import share_storage
import models
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
        actor = models.Actor()
        critic=models.Critic()
        actor_weigths = actor.get_weights()
        critic_weights=critic.get_weights()
        # summary = str(model).replace("\n", " \n\n")
        return actor_weigths, critic_weights

if __name__=="__main__":
    ray.init()
    checkpoint={
        "game":game_name,
        "actor_weights":None,
        "critic_weights":None,
        "lr":1e-4,
        "terminate":False,
        "start_training":False,
        "training_step":0,
        "max_training_step":2e3,
        "memory_update_iter":20,
        "max_len_episode":2e3,
        "num_update":10,
        "memory_size":2048,
        "batch_size":64,
        "epsilon":0.2,
        "gamma":0.95,
        "entropy_beta":1e-3,
        "epsilon":1,
        "replace_target_iter":1000,
        "tau":0.002,
        "max_len_step":10,
        "action_list":[0,2,3]
    }

    if os.path.exists(path_file):
        model_params=torch.load(path_file) 
        checkpoint["actor_weights"]=copy.deepcopy(model_params["actor_weights"])
        checkpoint["critic_weights"]=copy.deepcopy(model_params["critic_weights"])
        print('Load model!')

    else:
        cpu_actor=ModelLoader()
        actor_weights,critic_weights = cpu_actor.get_initial_weights()
        checkpoint["actor_weights"] = copy.deepcopy(actor_weights)
        checkpoint["critic_weights"]=copy.deepcopy(critic_weights)
        print('Init model!')

    # cpu_actor=ModelLoader()
    # actor_weights,critic_weights = cpu_actor.get_initial_weights()
    # checkpoint["actor_weights"] = copy.deepcopy(actor_weights)
    # checkpoint["critic_weights"]=copy.deepcopy(critic_weights)
    # print('Init model!')
 
    # init nodes
    share_storage_worker = share_storage.SharedStorage.remote(checkpoint)

    replay_buffer_worker = replay_buffer.ReplayBuffer.remote(checkpoint, share_storage_worker)
    
    training_worker=trainer.Trainer.options(num_cpus=1,num_gpus=1).remote(checkpoint,share_storage_worker,replay_buffer_worker)

    self_play_workers = [player.Player.options(num_cpus=1,num_gpus=0.1).remote(checkpoint,True,share_storage_worker,replay_buffer_worker,training_worker,False) 
                                        for _ in range(NUM_WORKER)]

    print("init nodes done!")
    

    # start works
    [worker.continous_self_play.remote() for worker in self_play_workers]
    training_worker.continous_update_weights.remote()

    while True:
        time.sleep(5)

    print('Done!')

