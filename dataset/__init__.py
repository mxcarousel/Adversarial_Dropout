# General Dependecies
import os

# PyTorch Dependencies
from torch.utils.data import DataLoader
import torchvision.transforms as tfs

from torchvision.datasets import CIFAR10, MNIST

from .distributed_dataset import distributed_dataset
from .distributed_dataset_givenclass import distributed_dataset_givenclass


class get_dataset:
    def __init__(self, dataset_name, num_clients, class_per_client, imbalance,
                batch_size=None,
                transforms=None, 
                seed=777, iid=1, path="../data", **kwargs): #No need of rank
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.transforms = transforms
        self.seed = seed
        self.iid = iid
        self.num_clients = num_clients
        self.class_per_client = class_per_client
        self.balance = 1-imbalance
        self.path = path
        if self.dataset_name == 'CIFAR10':
            if self.transforms is None:
                self.transforms = tfs.Compose([
                    tfs.ToTensor(),
                    tfs.Normalize([0.4915, 0.4823, 0.4468], 
                                  [0.247, 0.2435, 0.2616])
                ])
            if self.batch_size is None:
                self.batch_size = 1
            if not os.path.exists(path):
                os.mkdir(path)
            self.train_set = CIFAR10(root=path, train=True, download=True, 
                                transform=self.transforms)
            self.test_set = CIFAR10(root=path, train=False, download=True, 
                                transform=self.transforms)
            self.input_size = (3, 32, 32)
            self.classes = 10
        
        elif self.dataset_name == "MNIST":
            if self.transforms is None:
                self.transforms = tfs.Compose([
                            tfs.ToTensor(),
                            tfs.Normalize((0.1307,), (0.3081,))
                        ])
            if self.batch_size is None:
                self.batch_size = 1
            if not os.path.exists(path):
                os.mkdir(path)
            self.train_set = MNIST(root=path, train=True, download=True, transform=self.transforms)
            self.test_set = MNIST(root=path, train=False, download=True, transform=self.transforms)
            self.input_size = (1, 28, 28)
            self.classes = 10
        
        """"elif self,dataset_name == xxx"""


        # all set loading dataset
    def fetch_rank_data(self, rank, iid, is_test):
        statist = []
        if is_test:
            train_loader = DataLoader(self.train_set, batch_size = self.batch_size, drop_last=True, num_workers=4)
            test_loader = DataLoader(self.test_set, batch_size=self.batch_size, drop_last=True)
        else:
            if iid == 1:
                train_set, statist = distributed_dataset(self.train_set, rank, self.num_clients, balance=self.balance, seed = self.seed, num_classes=self.classes)
            elif iid == 2:
                train_set, statist = distributed_dataset_givenclass(self.train_set, rank, self.num_clients, class_per_client=self.class_per_client, balance=self.balance, seed = self.seed,classes=self.classes)
            train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True, drop_last=True)
            test_loader = DataLoader(self.test_set, batch_size=self.batch_size, drop_last=True)
        return train_loader, test_loader, self.input_size, self.classes, statist
