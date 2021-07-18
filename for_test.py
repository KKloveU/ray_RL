import os
import ray
import numpy as np
from ray import worker
import torch
import player
import trainer
import replay_buffer
import share_storage
import models
import models_atten
import models_dropout
import models_dueing
import models_dueing_dropout_atten
import models_dueing_noisy
import models_dueing_noisy_atten
import models_noisy
import copy
import time

NUM_WORKER = 1
game_name = "Breakout"
path_file = r"./model_checkpoint/"+game_name+".model"

    # play_worker_list=[]
    # train_worker_list=[]
    # share_worker_list=[]
    # replay_worker_list=[]

class CPUActor:
    # Trick to force DataParallel to stay on CPU to get weights on CPU even if there is a GPU
    def __init__(self):
        pass

    def get_initial_weights(self,test_model):
        model = test_model.Model()
        weigths = model.get_weights()
        summary = str(model).replace("\n", " \n\n")
        return weigths, summary



# def for_test(checkpoint,test_model,model_name):
#     share_storage_worker = share_storage.SharedStorage.remote(checkpoint,model_name)

#     replay_buffer_worker = replay_buffer.ReplayBuffer.remote(
#         checkpoint, share_storage_worker)

#     training_worker = trainer.Trainer.options(num_gpus=0.6).remote(
#         checkpoint, replay_buffer_worker, share_storage_worker,test_model)

#     self_play_workers = [player.Player.options(num_gpus=0.1).remote(
#         checkpoint,replay_buffer_worker,share_storage_worker, False,test_model,model_name) for _ in range(NUM_WORKER)]

#     self_play_workers.append(player.Player.options(
#         num_gpus=0.1).remote(checkpoint, replay_buffer_worker,share_storage_worker, True,test_model,model_name))

#     [worker.continous_self_play.remote() for worker in self_play_workers]
#     training_worker.continous_update_weights.remote()



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

    # for_test=[]

    model_name='_atten'
    test_model=models_atten

    cpu_actor = CPUActor()
    cpu_weights = cpu_actor.get_initial_weights(test_model)
    checkpoint["weights"], summary = copy.deepcopy(cpu_weights)
    share_storage_worker = share_storage.SharedStorage.remote(checkpoint,model_name)
    replay_buffer_worker = replay_buffer.ReplayBuffer.remote(
        checkpoint, share_storage_worker)
    training_worker = trainer.Trainer.options(num_gpus=0.6).remote(
        checkpoint, replay_buffer_worker, share_storage_worker,test_model)
    self_play_workers = [player.Player.options(num_gpus=0.1).remote(
        checkpoint,replay_buffer_worker,share_storage_worker, False,test_model,model_name) for _ in range(NUM_WORKER)]
    self_play_workers.append(player.Player.options(
        num_gpus=0.1).remote(checkpoint, replay_buffer_worker,share_storage_worker, True,test_model,model_name))
    [worker.continous_self_play.remote() for worker in self_play_workers]
    training_worker.continous_update_weights.remote()


    model_name='_dropout'
    test_model=models_dropout

    cpu_actor = CPUActor()
    cpu_weights = cpu_actor.get_initial_weights(test_model)
    checkpoint["weights"], summary = copy.deepcopy(cpu_weights)
    share_storage_worker1 = share_storage.SharedStorage.remote(checkpoint,model_name)
    replay_buffer_worker1 = replay_buffer.ReplayBuffer.remote(
        checkpoint, share_storage_worker1)
    training_worker = trainer.Trainer.options(num_gpus=0.6).remote(
        checkpoint, replay_buffer_worker1, share_storage_worker1,test_model)
    self_play_workers = [player.Player.options(num_gpus=0.1).remote(
        checkpoint,replay_buffer_worker1,share_storage_worker1, False,test_model,model_name) for _ in range(NUM_WORKER)]
    self_play_workers.append(player.Player.options(
        num_gpus=0.1).remote(checkpoint, replay_buffer_worker1,share_storage_worker1, True,test_model,model_name))
    [worker.continous_self_play.remote() for worker in self_play_workers]
    training_worker.continous_update_weights.remote()


    model_name='_dueing_drop_atten'
    test_model=models_dueing_dropout_atten

    cpu_actor = CPUActor()
    cpu_weights = cpu_actor.get_initial_weights(test_model)
    checkpoint["weights"], summary = copy.deepcopy(cpu_weights)
    share_storage_worker2 = share_storage.SharedStorage.remote(checkpoint,model_name)
    replay_buffer_worker2 = replay_buffer.ReplayBuffer.remote(
        checkpoint, share_storage_worker2)
    training_worker = trainer.Trainer.options(num_gpus=0.6).remote(
        checkpoint, replay_buffer_worker2, share_storage_worker2,test_model)
    self_play_workers = [player.Player.options(num_gpus=0.1).remote(
        checkpoint,replay_buffer_worker2,share_storage_worker2, False,test_model,model_name) for _ in range(NUM_WORKER)]
    self_play_workers.append(player.Player.options(
        num_gpus=0.1).remote(checkpoint, replay_buffer_worker2,share_storage_worker2, True,test_model,model_name))
    [worker.continous_self_play.remote() for worker in self_play_workers]
    training_worker.continous_update_weights.remote()


    model_name='_dueing_noisy_atten'
    test_model=models_dueing_noisy_atten

    cpu_actor = CPUActor()
    cpu_weights = cpu_actor.get_initial_weights(test_model)
    checkpoint["weights"], summary = copy.deepcopy(cpu_weights)
    share_storage_worker3 = share_storage.SharedStorage.remote(checkpoint,model_name)
    replay_buffer_worker3 = replay_buffer.ReplayBuffer.remote(
        checkpoint, share_storage_worker)
    training_worker = trainer.Trainer.options(num_gpus=0.6).remote(
        checkpoint, replay_buffer_worker3, share_storage_worker3,test_model)
    self_play_workers = [player.Player.options(num_gpus=0.1).remote(
        checkpoint,replay_buffer_worker3,share_storage_worker3, False,test_model,model_name) for _ in range(NUM_WORKER)]
    self_play_workers.append(player.Player.options(
        num_gpus=0.1).remote(checkpoint, replay_buffer_worker3,share_storage_worker3, True,test_model,model_name))
    [worker.continous_self_play.remote() for worker in self_play_workers]
    training_worker.continous_update_weights.remote()



    model_name='_dueing_noisy'
    test_model=models_dueing_noisy

    cpu_actor = CPUActor()
    cpu_weights = cpu_actor.get_initial_weights(test_model)
    checkpoint["weights"], summary = copy.deepcopy(cpu_weights)
    share_storage_worker4 = share_storage.SharedStorage.remote(checkpoint,model_name)
    replay_buffer_worker4 = replay_buffer.ReplayBuffer.remote(
        checkpoint, share_storage_worker)
    training_worker = trainer.Trainer.options(num_gpus=0.6).remote(
        checkpoint, replay_buffer_worker4, share_storage_worker4,test_model)
    self_play_workers = [player.Player.options(num_gpus=0.1).remote(
        checkpoint,replay_buffer_worker4,share_storage_worker4, False,test_model,model_name) for _ in range(NUM_WORKER)]
    self_play_workers.append(player.Player.options(
        num_gpus=0.1).remote(checkpoint, replay_buffer_worker4,share_storage_worker4, True,test_model,model_name))
    [worker.continous_self_play.remote() for worker in self_play_workers]
    training_worker.continous_update_weights.remote()


    model_name='_dueing'
    test_model=models_dueing

    cpu_actor = CPUActor()
    cpu_weights = cpu_actor.get_initial_weights(test_model)
    checkpoint["weights"], summary = copy.deepcopy(cpu_weights)
    share_storage_worker5 = share_storage.SharedStorage.remote(checkpoint,model_name)
    replay_buffer_worker5 = replay_buffer.ReplayBuffer.remote(
        checkpoint, share_storage_worker)
    training_worker = trainer.Trainer.options(num_gpus=0.6).remote(
        checkpoint, replay_buffer_worker5, share_storage_worker5,test_model)
    self_play_workers = [player.Player.options(num_gpus=0.1).remote(
        checkpoint,replay_buffer_worker5,share_storage_worker5, False,test_model,model_name) for _ in range(NUM_WORKER)]
    self_play_workers.append(player.Player.options(
        num_gpus=0.1).remote(checkpoint, replay_buffer_worker5,share_storage_worker5, True,test_model,model_name))
    [worker.continous_self_play.remote() for worker in self_play_workers]
    training_worker.continous_update_weights.remote()


    model_name='_noisy'
    test_model=models_noisy

    cpu_actor = CPUActor()
    cpu_weights = cpu_actor.get_initial_weights(test_model)
    checkpoint["weights"], summary = copy.deepcopy(cpu_weights)
    share_storage_worker6 = share_storage.SharedStorage.remote(checkpoint,model_name)
    replay_buffer_worker6 = replay_buffer.ReplayBuffer.remote(
        checkpoint, share_storage_worker)
    training_worker = trainer.Trainer.options(num_gpus=0.6).remote(
        checkpoint, replay_buffer_worker6, share_storage_worker6,test_model)
    self_play_workers = [player.Player.options(num_gpus=0.1).remote(
        checkpoint,replay_buffer_worker6,share_storage_worker6, False,test_model,model_name) for _ in range(NUM_WORKER)]
    self_play_workers.append(player.Player.options(
        num_gpus=0.1).remote(checkpoint, replay_buffer_worker6,share_storage_worker6, True,test_model,model_name))
    [worker.continous_self_play.remote() for worker in self_play_workers]
    training_worker.continous_update_weights.remote()


    model_name=''
    test_model=models

    cpu_actor = CPUActor()
    cpu_weights = cpu_actor.get_initial_weights(test_model)
    checkpoint["weights"], summary = copy.deepcopy(cpu_weights)
    share_storage_worker7 = share_storage.SharedStorage.remote(checkpoint,model_name)
    replay_buffer_worker7 = replay_buffer.ReplayBuffer.remote(
        checkpoint, share_storage_worker)
    training_worker = trainer.Trainer.options(num_gpus=0.6).remote(
        checkpoint, replay_buffer_worker7, share_storage_worker,test_model)
    self_play_workers = [player.Player.options(num_gpus=0.1).remote(
        checkpoint,replay_buffer_worker7,share_storage_worker7, False,test_model,model_name) for _ in range(NUM_WORKER)]
    self_play_workers.append(player.Player.options(
        num_gpus=0.1).remote(checkpoint, replay_buffer_worker7,share_storage_worker7, True,test_model,model_name))
    [worker.continous_self_play.remote() for worker in self_play_workers]
    training_worker.continous_update_weights.remote()



    while True:
        time.sleep(10)

    



        