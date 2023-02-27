from torch.optim import SGD

from models import get_model
from dataset import get_dataset
from backend import Worker
from backend import Server
from utils import cal_weight,np

from algorithm.amfed import amfed_trainer
from algorithm.amfed_prox import amfed_prox_trainer
from algorithm.fedavg import fedavg_trainer
from algorithm.fedprox import fedprox_trainer
from algorithm.mifa import mifa_trainer




def train(config, baseline_frac, server, worker_list, temp_train_loader, temp_test_loader, tb,stat):
    if config['algorithm'] == 'AMFED':
        trainer = amfed_trainer(worker_list, server,temp_train_loader, temp_test_loader, **config)
    elif config['algorithm'] == 'AMFED_Prox':
        trainer = amfed_prox_trainer(worker_list, server,temp_train_loader, temp_test_loader, **config)
    elif config['algorithm'] == 'FEDAVG':
        trainer = fedavg_trainer(worker_list, server,temp_train_loader, temp_test_loader, **config)
    elif config['algorithm'] == 'FEDAVG_Prox':
        trainer = fedprox_trainer(worker_list, server,temp_train_loader, temp_test_loader, **config)
    elif config['algorithm'] == 'MIFA':
        trainer = mifa_trainer(worker_list, server,temp_train_loader, temp_test_loader, **config)
    
    trainer.train(baseline_frac, config['sample_strategy'],\
                    config['communication_round'], \
                        config['lr'], config['algorithm'], tb, config['n_swap'], config['seed'], stat, config)



def dataset_init(iid,dataset_name,num_clients,class_per_client,imbalance,batch_size,seed,path):

    dataset_class = get_dataset(iid = iid,
                            dataset_name=dataset_name,
                            num_clients=num_clients,
                            class_per_client=class_per_client,
                            imbalance = imbalance,
                            batch_size=batch_size,
                            seed=seed,
                            path=path)

    temp_train_loader, temp_test_loader, _, _, _ = dataset_class.fetch_rank_data(0, iid,1)

    return dataset_class, temp_train_loader, temp_test_loader

def worker_init(size,dataset_class,model_name,weight_decay,lr,gpu,iid):
    worker_list = []
    num_step = []

    for rank in range(size):
        train_loader, test_loader, input_size, classes, stat = dataset_class.fetch_rank_data(rank, iid, 0)
        model = get_model(model_name, input_size, classes)
        model = model.cuda()
        optimizer = SGD(model.parameters(), lr=lr, momentum=0, weight_decay=weight_decay)
        worker = Worker(rank=rank, model=model,
                        train_loader=train_loader, test_loader=test_loader,
                        optimizer=optimizer, lr=lr, gpu=gpu)
        worker.update_iter()
        worker_list.append(worker)
        num_step.append(train_loader.__len__())
    """Server initialization, split = weight"""
    with open('stat.npy', 'wb') as f:
        np.save(f, stat)
    weight = cal_weight(stat)

    return worker_list, input_size, classes, weight, stat, num_step

def server_init(model_name, input_size, classes, algorithm, weight, sample_frac):
    
    global_model = get_model(model_name, input_size, classes)
    global_model = global_model.cuda()
    server = Server(global_model=global_model, algorithm=algorithm, weight=weight, prob = sample_frac, gpu=True)

    return server

   

