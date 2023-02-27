from .uar import uar
from .prob import prob

"""active_clients is a binary indicator"""
"""selected_clients is the index of the selected clients."""

def sample_client(sample_strategy, size, frac):
    
    if sample_strategy == 'uar':
        active_clients, selected_clients = uar(size, frac)
    elif sample_strategy == 'withp':
        active_clients, selected_clients = prob(size, frac)
    
    return active_clients, selected_clients