import torch
import math

def uar(size, frac):
    baseline_size = int(math.floor( frac * size ))
    active_clients = torch.randperm(size)[:baseline_size]
    zeros_clients = torch.zeros(size)
    zeros_clients[active_clients] = 1
    active_clients = zeros_clients
    selected_clients = torch.arange(size)[active_clients==1]
    return active_clients, selected_clients