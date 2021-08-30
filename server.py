import ray
import torch

@ray.remote
class Server:
    def __init__(self,checkpoint,play_worker_list,train_worker) -> None:
        
        self.play_worker_list=play_worker_list
        self.train_worker=train_worker
        self.weights=checkpoint['weights']

    def run(self):
        while True:

            workers=[worker.continous_self_play.remote(self.weights) for worker in self.play_worker_list]
            while len(workers):
                done,workers=ray.wait(workers)
            # print('#########')
            self.weights=ray.get(self.train_worker.update.remote())
