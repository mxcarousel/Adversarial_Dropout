import numpy as np
from torch.nn import Module
import copy
import torch

def model_initialization(model):
    temp_model = copy.deepcopy(model)
    for name, params in temp_model.named_parameters():
        params.data = torch.zeros_like(params.data)
    return temp_model

class Server:
    def __init__(self, global_model: Module, algorithm: str, weight: list, prob: float, gpu=True):
        super().__init__()
        self.global_model = global_model
        self.G = model_initialization(global_model)
        self.weight = weight
        self.algorithm = algorithm
        self.global_model.cuda()
        self.G.cuda()
        self.prob = prob

    def get_model_params(self):
        state_dict = self.global_model.state_dict()
        return state_dict