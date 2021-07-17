import os
import ray
import numpy as np
import torch
import player
import trainer
import share_storage
import models
import copy
import time

NUM_WORKER = 10
game_name = "Pong"
path_file = r"./model_checkpoint/"+game_name+".model"


class ModelLoader:
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
        "gamma": 0.99,
        "model_save_iter": 100,
        "memory_update_iter":20,
        "max_len_step": 10,
        "action_list": [0, 2, 3]
    }

    if os.path.exists(path_file):
        model_params = torch.load(path_file)
        checkpoint["weights"] = copy.deepcopy(model_params["weights"])
        print('Load model!')
    else:
        cpu_actor = ModelLoader()
        cpu_weights = cpu_actor.get_initial_weights()
        checkpoint["weights"], summary = copy.deepcopy(cpu_weights)
        print('Init model!')

    # init nodes
    share_storage_worker = share_storage.SharedStorage.options().remote(checkpoint)

    training_worker = trainer.Trainer.options(
        num_gpus=0.1).remote(checkpoint, share_storage_worker)

    self_play_workers = [player.Player.options(num_gpus=0.1).remote(checkpoint, share_storage_worker, training_worker, False)
                         for _ in range(NUM_WORKER)]

    self_play_workers.append(player.Player.options(num_gpus=0.1).remote(
        checkpoint, share_storage_worker, training_worker, True))

    print("init nodes done!")

    # start works
    [worker.continous_self_play.remote() for worker in self_play_workers]

    while True:
        time.sleep(5)

    print('Done!')
