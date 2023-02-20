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

    def aggregation(self, selected_clients, packages,speedup):
        self.global_model.train()
        if self.algorithm == 'AMFED' or self.algorithm == 'AMFED_Prox':
            selected_clients = selected_clients.reshape(-1)
            weight_tensor = torch.tensor(self.weight)
            selected_weight = weight_tensor[selected_clients].sum()
            if speedup:
                su = 1/np.sqrt(self.prob)
                for name, param in self.global_model.named_parameters():
                    param.data *= (1-selected_weight)+selected_weight*(1-su)
                    for index in selected_clients:
                        param.data += su*packages[index].get_model_params()[name].data * self.weight[index]
            else:
                for name, param in self.global_model.named_parameters():
                    param.data *= (1-selected_weight)
                    for index in selected_clients:
                        param.data += packages[index].get_model_params()[name].data * self.weight[index]
        
        elif self.algorithm == 'FEDAVG' or self.algorithm == 'FEDAVG_Prox':
            selected_clients = selected_clients.reshape(-1)
            weight_tensor = torch.tensor(self.weight)
            selected_weight = weight_tensor[selected_clients].sum()
            weight_tensor /= selected_weight # renormalization
            for name, param in self.global_model.named_parameters():
                param.data *= 0
                for index in selected_clients:
                    param.data += packages[index].get_model_params()[name].data * weight_tensor[index]
        
        elif self.algorithm == 'MIFA':
            selected_clients = selected_clients.reshape(-1)
            weight_tensor = torch.tensor(self.weight)
            selected_weight = weight_tensor[selected_clients].sum()
            index = 0
            for index in selected_clients:
                for name, param in packages[index].G1.named_parameters():
                    param.data = (self.get_model_params()[name].data -packages[index].get_model_params()[name].data)/packages[index].get_lr()
                for name, param in self.G.named_parameters():
                    param.data += (packages[index].get_G1_params()[name].data - 
                                   packages[index].get_G0_params()[name].data) * weight_tensor[index]
                packages[index].load_G0_params(packages[index].get_G1_params())
            for name, param in self.global_model.named_parameters():
                param.data -= self.G.state_dict()[name].data * packages[index].get_lr()
        
    
    def get_model_params(self):
        state_dict = self.global_model.state_dict()
        return state_dict