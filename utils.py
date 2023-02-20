from collections import OrderedDict
import torch
import numpy as np
import random

def state_dic_divide(dict,constant):
    temp=OrderedDict()
    for name, param in dict.items():
        temp[name] = param.data /constant
    return temp

def state_dic_sum(original, new):
    temp = OrderedDict()
    for name, param in original.items():
        temp[name] = new[name].data + param.data
    return temp

def state_dic_sub(former, latter, P, i):
    temp_dict = OrderedDict()
    for name, param in former[i][0].items():
        temp_dict[name] = torch.zeros_like(param.data)
        for j in range(len(former)):
            if P[j,i]:
                temp_dict[name] += former[i][j][name].data-latter[i][j][name].data
    return temp_dict


def drop_out_forever(alpha: float, stat):
    size = stat.shape[0]
    nums_of_selection = np.floor(float(alpha)* size).astype(int)
    stat_sort_ind = np.argsort(stat.sum(axis=1))
    selected_client_index = np.arange(size)[stat_sort_ind>size-nums_of_selection]
    selected_client_tensor = torch.from_numpy(selected_client_index.copy())
    return selected_client_tensor

def drop_out_deter(active_list: torch.tensor, deter_list: torch.tensor):
    active_list = torch.arange(len(active_list))[active_list==1]
    concat_list = torch.cat((active_list, deter_list))
    selected_client_index = torch_delete(concat_list, deter_list)#retain[retain_count==1]
    return selected_client_index

def drop_out_alternating_slice(repetition: int, stat):
    np.random.seed(123)
    worker_list = np.arange((np.array(stat)).shape[0])
    np.random.shuffle(worker_list)
    worker_index = np.split(worker_list,repetition)
    return worker_index

def drop_out_alternating(active_list: torch.tensor, interval: int, repetition: int, round: int, worker_index):
    round -= 1
    slice_index = 0
    # worker index ready for each repetition
    if not round % interval:
        slice_index = (round // interval) % repetition 
    #slice index denote the repetition threshold
    selected_client_index = drop_out_deter(active_list, torch.tensor(worker_index[slice_index]))
    return selected_client_index


def unbalanced_split(dataset, size):
    np.random.seed(123)
    """training set size: 5*1e4 for CIFAR10"""
    """                   6*1e4 for MNIST  """
    img_size = len(dataset)
    # if dataset_name == 'CIFAR10':
    #     img_size = 50000
    # elif dataset_name == 'MNIST':
    #     img_size = 60000
    mu, sigma = img_size / size, 10000 / size
    split = np.random.normal(mu, sigma, size)/img_size
    if split.sum()>1:
        split[np.argmax(split)] -= split.sum()-1
    return split

def init_deterministic(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)

def cal_weight(stat):
    return (stat.sum(axis=1)/stat.sum(axis=1).sum()).tolist()

def torch_delete(tensor, list):
    mask = torch.ones(tensor.numel(), dtype=torch.bool)
    for i in list:
        mask[tensor==i] = False
    return tensor[mask]