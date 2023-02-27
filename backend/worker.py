from typing import List

from torch.utils.data import DataLoader
from torch.nn import Module
import torch.nn as nn
import copy

import torch

def model_initialization(model):
    temp_model = copy.deepcopy(model)
    for name, params in temp_model.named_parameters():
        params.data = torch.zeros_like(params.data)
    return temp_model

class Worker:
    def __init__(self, rank, model: Module,
                 train_loader: DataLoader, test_loader: DataLoader,
                 optimizer, lr,
                 gpu=True):
        self.rank = rank
        self.model = model
        self.G0 = model_initialization(model)
        self.G1 = model_initialization(model)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.train_loader_iter = train_loader.__iter__()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optimizer
        self.lr = lr
        self.model.cuda()
        self.G0.cuda()
        self.G1.cuda()

    def update_iter(self):
        self.train_loader_iter = self.train_loader.__iter__()

    def wakeup_step(self,constant):
        self.model.train()
        self.model.cuda()

        batch = self.train_loader_iter.__next__()
        data, target = batch[0].cuda(), batch[1].cuda()
        output = self.model(data)
        loss = self.criterion(output, target)
        self.optimizer.zero_grad()
        self.optimizer.param_groups[0]['lr'] *= constant
        loss.backward()

    def step(self):
        self.model.train()
        self.model.cuda()
        batch = self.train_loader_iter.__next__()
        data, target = batch[0].cuda(), batch[1].cuda()
        output = self.model(data)
        loss = self.criterion(output, target)
        self.optimizer.zero_grad()
        loss.backward()

    
    def fedprox_step(self, global_model, mu):
        self.model.train()
        self.model.cuda()
        global_model.cuda()
        batch = self.train_loader_iter.__next__()
        data, target = batch[0].cuda(), batch[1].cuda()
        self.optimizer.zero_grad()
        output = self.model(data)
        proximal_term = 0.0
        for w, w_t in zip(self.model.parameters(), global_model.parameters()):
            proximal_term += (w - w_t).norm(2)
        loss = self.criterion(output, target)+(mu / 2) * proximal_term
        loss.backward()

    def update_grad(self):
        self.optimizer.step()

    def update_grad_decay(self, lr):
        for g in self.optimizer.param_groups:
            g['lr'] = lr
        self.optimizer.step()
    
    def get_lr(self):
        return self.lr

    def load_model_params(self, state_dict):
        self.model.load_state_dict(state_dict)
    
    def get_model_params(self):
        self.model.cuda()
        state_dict = self.model.state_dict()
        return state_dict

    def get_G0_params(self):
        self.G0.cuda()
        state_dict = self.G0.state_dict()
        return state_dict

    def get_G1_params(self):
        self.G1.cuda()
        state_dict = self.G1.state_dict()
        return state_dict

    def load_G0_params(self, state_dict):
        self.G0.load_state_dict(state_dict)
    
    def model_cpu(self):
        self.model.to('cpu')
        self.G0.to('cpu')
        self.G1.to('cpu')