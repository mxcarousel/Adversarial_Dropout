from torch.utils.data import Dataset
from typing import Tuple, Any
import torch
import random
from utils import *


class DistributedDataset(Dataset):
    def __init__(self, dataset: Dataset, index):
        super().__init__()
        self.dataset = dataset
        self.index = index

    def __getitem__(self, item):
        return self.dataset.__getitem__(self.index[item])

    def __len__(self):
        return len(self.index)

def distributed_dataset(dataset: Dataset, rank: int, num_clients, balance=0, seed: int = 777,num_classes=10):
    size = len(dataset)
    random.seed(seed)
    indexes = [x for x in range(size)]
    random.shuffle(indexes)
    indexes_list, y, statistic = [],[[] for _ in range(num_clients)],np.zeros((num_clients,num_classes))

    dataset.targets = np.array(dataset.targets)

    if balance:
        split = [1.0 / size for _ in range(num_clients)]
    else:
        split = unbalanced_split(dataset, num_clients)
    
    for s in split:
        indexes_list.append(indexes[:int(s * size)])
        indexes = indexes[int(s * size):]

    for client in range(num_clients):
        idxs = indexes_list[client]
        y[client] = dataset.targets[idxs]
        for i in np.unique(y[client]):
            statistic[client][int(i)]=int(sum(y[client]==i))

    return DistributedDataset(dataset, indexes_list[rank]), statistic
