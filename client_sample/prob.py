import torch
# frac is a single constant

def prob(size, frac):
    active_clients = torch.bernoulli(frac * torch.ones(size))
    selected_clients = torch.arange(size)[active_clients==1]
    return active_clients, selected_clients