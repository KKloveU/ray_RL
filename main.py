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
model='_dueing_noisy_atten'
path_file = r"./model_checkpoint/"+game_name+model+".model"

class CPUActor:
    # Trick to force DataParallel to stay on CPU to get weights on CPU even if there is a GPU
    def __init__(self):
        pass

    def get_initial_weights(self):
        model = models.Model()
        weigths = model.get_weights()
        summary = str(model).replace("\n", " \n\n")
        return weigths, summary


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

    if os.path.exists(path_file):
        model_params = torch.load(path_file)
        checkpoint["weights"] = copy.deepcopy(model_params["weights"])
        print('Load model!')
    else:
        cpu_actor = CPUActor()
        cpu_weights = cpu_actor.get_initial_weights()
        checkpoint["weights"], summary = copy.deepcopy(cpu_weights)
        print('Init model!')

    # init nodes
    share_storage_worker = share_storage.SharedStorage.remote(checkpoint,model)

    replay_buffer_worker = replay_buffer.ReplayBuffer.remote(
        checkpoint, share_storage_worker)

    training_worker = trainer.Trainer.options(num_gpus=0.1).remote(
        checkpoint, replay_buffer_worker, share_storage_worker)

    self_play_workers = [player.Player.options(num_gpus=0.6).remote(
        checkpoint,replay_buffer_worker,share_storage_worker, False,model) for _ in range(NUM_WORKER)]

    self_play_workers.append(player.Player.options(
        num_gpus=0.1).remote(checkpoint, replay_buffer_worker,share_storage_worker, True,model))

    print("init nodes done!")

    # start works
    [worker.continous_self_play.remote() for worker in self_play_workers]
    training_worker.continous_update_weights.remote()

    while True:
        time.sleep(5)

    print('Done!')
