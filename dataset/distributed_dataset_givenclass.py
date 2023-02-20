from torch.utils.data import Dataset
import numpy as np


class DistributedDataset(Dataset):
    def __init__(self, dataset: Dataset, index):
        super().__init__()
        self.dataset = dataset
        self.index = index

    def __getitem__(self, item):
        return self.dataset.__getitem__(self.index[item])

    def __len__(self):
        return len(self.index)

def distributed_dataset_givenclass(dataset: Dataset, rank: int, num_clients: int = None, class_per_client = 1, balance = 0, seed: int = 777, classes=10):
    num_classes = classes
    y = [[] for _ in range(num_clients)]
    statistic = np.zeros((num_clients,num_classes))
    dataidx_map = {}

    idxs = np.arange(len(dataset.targets))
    dataset.targets=np.array(dataset.targets)
    idx_for_each_class = []
    for i in range(num_classes):
        idx_for_each_class.append(idxs[dataset.targets == i])
    class_num_per_client = [class_per_client for _ in range(num_clients)]
    for i in range(num_classes):
        selected_clients = []
        for client in range(num_clients):
            if class_num_per_client[client] > 0:
                selected_clients.append(client)
            selected_clients = selected_clients[:int(num_clients/num_classes*class_per_client)]
        ### num_per = #_per_class_pic/ num_selected_clients
        num_all_samples = len(idx_for_each_class[i])
        num_selected_clients = len(selected_clients)
        num_per = num_all_samples / num_selected_clients
        if balance:
            num_samples = [int(num_per) for _ in range(num_selected_clients-1)]
        else:
            mu = num_per
            sigma = num_per/num_all_samples*1e3
            num_samples = np.floor(np.random.normal(mu, sigma, num_selected_clients)).astype(int)
            if num_samples.sum()>num_all_samples:
                num_samples = np.floor(num_samples/num_samples.sum()*num_all_samples).astype(int)
        idx = 0
        for client, num_sample in zip(selected_clients, num_samples):
            if client not in dataidx_map.keys():
                dataidx_map[client] = idx_for_each_class[i][idx:idx+num_sample]
            else:
                dataidx_map[client] = np.append(dataidx_map[client], idx_for_each_class[i][idx:idx+num_sample], axis=0)
            idx += num_sample
            class_num_per_client[client] -= 1 # the most important statement to filter the processed client!
        
        for client in selected_clients:
            idxs = dataidx_map[client]
            y[client] = dataset.targets[idxs]
            for i in np.unique(y[client]):
                statistic[client][int(i)]=int(sum(y[client]==i))
    return DistributedDataset(dataset, list(dataidx_map[rank])),statistic