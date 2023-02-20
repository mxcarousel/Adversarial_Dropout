import datetime
import os
import math
import torch
from torch import nn

from torch.utils.tensorboard import SummaryWriter
from torch.optim import SGD

from models import get_model
from dataset import get_dataset
from backend import Worker
from backend import Server
from utils import *
from config import get_config
from tqdm import tqdm

TEST_ACCURACY = 0


def work():
    config = get_config()
    print(config)
    run(**config)


def run(
        communication_round, model_name, dataset_name, class_per_client, size, repetition, interval,
        batch_size, lr, weight_decay, gpu, iid, step_decay,speedup,
        seed, alpha, imbalanced, dropout, local_steps, local_epochs, sample_frac,
        n_swap, path, mu, algorithm, **kwargs):
    tb = SummaryWriter(comment=f"seed_{seed}_model_{model_name}_dataset_{dataset_name}_bs_{batch_size}_lr_{lr}_size_{size}_alg_{algorithm}_sample_{sample_frac:g}_alpha_{alpha:g}_le_{local_epochs}_mu_{mu:g}_imbalan_{imbalanced}_iid_{iid}_sd_{step_decay}_do_{dropout}_rep_{repetition}_inter_{interval}_class_{class_per_client}_speedup_{speedup:g}")
    init_deterministic(seed)
     # temp_train, temp_test = whole set train loader/ whole set test loader
    criterion = nn.CrossEntropyLoss()
    worker_list = []
    num_step = []
    
    dataset_class = get_dataset(iid = iid,
                            dataset_name=dataset_name,
                            num_clients=size,
                            class_per_client=class_per_client,
                            imbalance = imbalanced,
                            batch_size=batch_size,
                            seed=seed,
                            path=path,
                            **kwargs)
    temp_train_loader, temp_test_loader, input_size, classes, _ = dataset_class.fetch_rank_data(0, iid,1)

    for rank in range(size):
        train_loader, test_loader, input_size, classes, stat = dataset_class.fetch_rank_data(rank, iid, 0)
        model = get_model(model_name, input_size, classes)
        model = model.cuda()
        optimizer = SGD(model.parameters(), lr=lr, momentum=0, weight_decay=weight_decay)
        worker = Worker(rank=rank, model=model,
                        train_loader=train_loader, test_loader=test_loader,
                        optimizer=optimizer, lr=lr, gpu=gpu)
        worker.update_iter() # Client's iterator has already been initialized.
        worker_list.append(worker)
        num_step.append(train_loader.__len__())
    """Server initialization, split = weight"""
    with open('stat.npy', 'wb') as f:
        np.save(f, stat)
    weight = cal_weight(stat)
    global_model = get_model(model_name, input_size, classes)
    global_model = global_model.cuda()
    server = Server(global_model=global_model, algorithm=algorithm, weight=weight, prob = sample_frac, gpu=True)
    print(f"| num_step: {num_step}")
    baseline = int(math.floor(sample_frac*size))
    """Adversarial drop out, where each client will be selected randomly at uniform."""
    """However, an arbitrary $alpha$ fraction of the selected clients will drop out."""
    """The first building block is that we need asynchronous local computations possibly s steps."""
    baseline_size = [size if i<5 else baseline for i in range(communication_round)]
    total_step = 0
    """The following statement defines the local-step as 2*the smallest number of batches."""
    if local_steps == 0:
        local_steps = np.min(local_epochs * np.array(num_step))
    """Initialize a temporary step-size."""
    lr0 = lr
    if dropout == 1:
        worker_slice = drop_out_alternating_slice(repetition, stat)
        repetition = len(worker_slice)
    elif dropout == 2:
        deter_clients = drop_out_forever(alpha,stat)

    if algorithm == 'AMFED_Prox' or algorithm == 'FEDAVG_Prox':
        for round in tqdm(range(1, communication_round + 1)):
            torch.random.manual_seed(round+seed)
            diff_list = []
            start = datetime.datetime.now()
            """ The following code is to 1) generate client sequence with probability $p$ 
                                        2) drop out $alpha$-fraction agents uniformly at random"""
            active_clients = torch.randperm(size)[:baseline_size[round-1]]
            zeros_clients = torch.zeros(size)
            zeros_clients[active_clients] = 1
            active_clients = zeros_clients
            selected_clients = torch.arange(size)[active_clients==1]

            if round > 5:
                if dropout == 1:
                    selected_clients = drop_out_alternating(active_clients, interval, repetition, round, worker_slice)
                elif dropout == 2:
                    selected_clients = drop_out_deter(active_clients, deter_clients)
            """"Only the working clients will compute $s$ local steps."""
            for index in selected_clients:
                worker = worker_list[index]
                worker.update_difference(server.global_model)
                worker.load_model_params(server.get_model_params())
                diff_list.append(worker.difference)
            """Now do $s$ steps local training at the chosen client"""
            for index in selected_clients:
                worker = worker_list[index]
                for step in range(local_steps):
                    total_step += 1
                    try:
                        worker.fedprox_step(server.global_model, mu)
                    except StopIteration:
                        worker.update_iter()
                        worker.fedprox_step(server.global_model, mu)
                    if step_decay:
                        worker.update_grad_decay(lr)
                    else:
                        worker.update_grad()
            """Time for global aggregation"""
            server.aggregation(selected_clients=selected_clients, packages=worker_list)
            if step_decay:
                lr = lr0 / np.sqrt(round)

            if round % 2 == 0:
                test_all(server.global_model, temp_train_loader, temp_test_loader,
                        criterion, None, total_step, tb, n_swap=n_swap)
            
            end = datetime.datetime.now()
            print(f"\r| Train | communication_round: {round}|{communication_round}, time: {(end - start).seconds}s",
                flush=True, end="")
        
    else:
        for round in tqdm(range(1, communication_round + 1)):
            torch.random.manual_seed(round+seed)
            diff_list = []
            start = datetime.datetime.now()
            active_clients = torch.randperm(size)[:baseline_size[round-1]]
            zeros_clients = torch.zeros(size)
            zeros_clients[active_clients] = 1
            active_clients = zeros_clients
            selected_clients = torch.arange(size)[active_clients==1]
            if round > 5:
                if dropout == 1:
                    selected_clients = drop_out_alternating(active_clients, interval, repetition, round, worker_slice)
                elif dropout == 2:
                    selected_clients = drop_out_deter(active_clients, deter_clients)
                
            """"Only the working clients will compute $s$ local steps."""
            for index in selected_clients:
                worker = worker_list[index]
                worker.update_difference(server.global_model)
                worker.load_model_params(server.get_model_params())
                diff_list.append(worker.difference)
                
            for index in selected_clients:
                worker = worker_list[index]
                """Now do $s$ steps local training at the chosen client"""
                for step in range(local_steps):
                    total_step += 1
                    try:
                        worker.step()
                    except StopIteration:
                        worker.update_iter()
                        worker.step()
                    if step_decay:
                        worker.update_grad_decay(lr)
                    else:
                        worker.update_grad()
            """We first consider a simpler scenario: balanced clients' data volume"""
            """Time for global aggregation"""
            server.aggregation(selected_clients=selected_clients, packages=worker_list,speedup=speedup)
            if step_decay and (algorithm == 'AMFED' or algorithm == 'FEDAVG'):
                lr = lr0 / np.sqrt(round)

            if round % 2 == 0:
                test_all(server.global_model, temp_train_loader, temp_test_loader,
                        criterion, round, total_step, tb,  n_swap=n_swap,algorithm=algorithm,speedup=speedup)
            
            end = datetime.datetime.now()
            print(f"\r| Train | communication_round: {round}|{communication_round}, time: {(end - start).seconds}s",
                flush=True, end="")


def test_all(model, train_loader, test_loader, criterion, epoch, total_step, tb, n_swap=None,algorithm='AMFED',speedup=0):
    print(f"\n| Test All |", flush=True, end="")
    model.cuda()
    model.eval()
    total_loss, total_correct, total, step = 0, 0, 0, 0
    start = datetime.datetime.now()
    for batch in train_loader:
        step += 1
        data, target = batch[0].cuda(), batch[1].cuda()
        output = model(data)
        p = torch.softmax(output, dim=1).argmax(1)
        total_correct += p.eq(target).sum().item()
        total += len(target)
        loss = criterion(output, target)
        total_loss += loss.item()
        end = datetime.datetime.now()
        print(f"\r| Test All |step: {step}, time: {(end - start).seconds}s", flush=True, end="")
    total_train_loss = total_loss / step
    total_train_acc = total_correct / total

    print(f'\n| Test All Train Set |'
            f' communication round: {epoch},'
            f' loss: {total_train_loss:.4},'
            f' acc: {total_train_acc:.4%}', flush=True)

    total_loss, total_correct, total, step = 0, 0, 0, 0

    for batch in test_loader:
        step += 1
        data, target = batch[0].cuda(), batch[1].cuda()
        output = model(data)
        p = torch.softmax(output, dim=1).argmax(1)
        total_correct += p.eq(target).sum().item()
        total += len(target)
        loss = criterion(output, target)
        total_loss += loss.item()
        end = datetime.datetime.now()
        print(f"\r| Test All |step: {step}, time: {(end - start).seconds}s", flush=True, end="")
    total_test_loss = total_loss / step
    total_test_acc = total_correct / total
    
    print(f'\n Algorithm {algorithm} SU{speedup}'
              f'\n| Test All Test Set |'
              f' communication round: {epoch},'
              f' loss: {total_test_loss:.4},'
              f' acc: {total_test_acc:.4%}', flush=True)

    tb.add_scalar("test loss", total_test_loss, epoch)
    tb.add_scalar("train loss", total_train_loss, epoch)
    tb.add_scalar("test acc", total_test_acc, epoch)
    tb.add_scalar("train acc", total_train_acc, epoch)

    if n_swap is not None:
        if not os.path.exists("./trained/"):
            os.mkdir("./trained/")
        if total_test_acc > TEST_ACCURACY:
            torch.save(model.state_dict(), f"./trained/{algorithm}_{n_swap}_best.pt")
        torch.save(model.state_dict(), f"./trained/{algorithm}_{n_swap}_last.pt")

if __name__ == '__main__':
    work()
