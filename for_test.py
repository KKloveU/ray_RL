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

NUM_WORKER = 1
game_name = "Breakout"
path_file = r"./model_checkpoint/"+game_name+".model"


class CPUActor:
    # Trick to force DataParallel to stay on CPU to get weights on CPU even if there is a GPU
    def __init__(self):
        pass

    def get_initial_weights(self):
        model = models.Model()
        weigths = model.get_weights()
        summary = str(model).replace("\n", " \n\n")
        return weigths, summary

@ray.remote(nums_gpu=1)
class For_test():
    def __init__(self,checkpoint,model) -> None:
        share_storage_worker = share_storage.SharedStorage.remote(checkpoint,model)

        replay_buffer_worker = replay_buffer.ReplayBuffer.remote(
            checkpoint, share_storage_worker)

        self.training_worker = trainer.Trainer.options(num_gpus=0.1).remote(
            checkpoint, replay_buffer_worker, share_storage_worker)

        self.self_play_workers = [player.Player.options(num_gpus=0.6).remote(
            checkpoint,replay_buffer_worker,share_storage_worker, False,model) for _ in range(NUM_WORKER)]

        self.self_play_workers.append(player.Player.options(
            num_gpus=0.1).remote(checkpoint, replay_buffer_worker,share_storage_worker,model, True))
    
    def run(self):
        [worker.continous_self_play.remote() for worker in self.self_play_workers]

        self.training_worker.continous_update_weights.remote()
        


if __name__ == "__main__":
    ray.init()
    checkpoint = {
        "game": game_name,
        "weights": None,
        "lr": 1e-4,
        "terminate": False,
        "start_training": False,
        "training_step": 0,
        "max_training_step": 2e3,
        "max_len_episode": 2e3,
        "memory_size": 5e3,
        "batch_size": 128,
        "gamma": 0.99,
        "epsilon": 1,
        "replace_target_iter": 100,
        "update_memory_iter": 200,
        "tau": 0.002,
        "episilon_decay": 1e-2,
        "action_list": [0, 2, 3]
    }

    for_test=[]

    model='_atten'
    cpu_actor = CPUActor()
    cpu_weights = cpu_actor.get_initial_weights()
    checkpoint["weights"], summary = copy.deepcopy(cpu_weights)
    for_test.append(For_test(checkpoint,model))


    model='_dropout'
    cpu_actor = CPUActor()
    cpu_weights = cpu_actor.get_initial_weights()
    checkpoint["weights"], summary = copy.deepcopy(cpu_weights)
    for_test.append(For_test(checkpoint,model))


    model='_dueing_drop_atten'
    cpu_actor = CPUActor()
    cpu_weights = cpu_actor.get_initial_weights()
    checkpoint["weights"], summary = copy.deepcopy(cpu_weights)
    for_test.append(For_test(checkpoint,model))


    model='_dueing_noisy_atten'
    cpu_actor = CPUActor()
    cpu_weights = cpu_actor.get_initial_weights()
    checkpoint["weights"], summary = copy.deepcopy(cpu_weights)
    for_test.append(For_test(checkpoint,model))


    model='_dueing_noisy'
    cpu_actor = CPUActor()
    cpu_weights = cpu_actor.get_initial_weights()
    checkpoint["weights"], summary = copy.deepcopy(cpu_weights)
    for_test.append(For_test(checkpoint,model))


    model='_dueing'
    cpu_actor = CPUActor()
    cpu_weights = cpu_actor.get_initial_weights()
    checkpoint["weights"], summary = copy.deepcopy(cpu_weights)
    for_test.append(For_test(checkpoint,model))


    model='_noisy'
    cpu_actor = CPUActor()
    cpu_weights = cpu_actor.get_initial_weights()
    checkpoint["weights"], summary = copy.deepcopy(cpu_weights)
    for_test.append(For_test(checkpoint,model))


    model=''
    cpu_actor = CPUActor()
    cpu_weights = cpu_actor.get_initial_weights()
    checkpoint["weights"], summary = copy.deepcopy(cpu_weights)
    for_test.append(For_test(checkpoint,model))

    [runner.run.remote() for runner in for_test]

    while True:
        time.sleep(10)

    



        