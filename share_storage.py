import copy
import ray
import torch

@ray.remote
class SharedStorage:
    def __init__(self,checkpoint) -> None:
        self.current_checkpoint=copy.deepcopy(checkpoint)
        self.path='./model_checkpoint/'+self.current_checkpoint["game"]+'_model'

    def save_checkpoint(self):
        torch.save(self.current_checkpoint,self.path)
    
    def get_checkpoint(self):
        return copy.deepcopy(self.current_checkpoint)

    def get_info(self, keys):
        if isinstance(keys, str):
            return self.current_checkpoint[keys]
        elif isinstance(keys, list):
            return {key: self.current_checkpoint[key] for key in keys}
        else:
            raise TypeError

    def set_info(self, keys, values=None):
        if isinstance(keys, str) and values is not None:
            self.current_checkpoint[keys] = values
        elif isinstance(keys, dict):
            self.current_checkpoint.update(keys)
        else:
            raise TypeError