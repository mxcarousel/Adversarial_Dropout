import datetime
import torch
import os
from torch import nn

from tqdm import tqdm

from utils import *
import sys

from drop_out_scheme.forever import forever_scheme
from drop_out_scheme.alternate import alternate_scheme

from abc import ABC, abstractmethod

from client_sample import sample_client
# from drop_out_scheme import dropout_init

class trainer(ABC):
    def __init__(self,worker_list, server, temp_train_loader, temp_test_loader, local_steps,step_decay,speedup,mu,**kwargs):
        self.temp_train_loader = temp_train_loader
        self.temp_test_loader = temp_test_loader
        self.worker_list = worker_list
        self.server = server
        self.local_steps = local_steps
        self.speedup = speedup
        self.step_decay = step_decay
        self.mu = mu
        super().__init__()

    def train(self, baseline_frac, sample_strategy, communication_round, \
                lr, algorithm, tb, n_swap, seed, stat, config):
        lr0 = lr
        criterion = nn.CrossEntropyLoss()
        dropout_class = dropout_init(stat,**config)
        for round in tqdm(range(1, communication_round + 1)):
            torch.random.manual_seed(round+seed)
            start = datetime.datetime.now()
            frac = baseline_frac[round-1]
            active_clients , selected_clients = sample_client(sample_strategy, len(self.worker_list), frac)

            selected_clients = dropout_class.drop(active_clients,selected_clients,round)
            """self.update let workers update their local models ($s$ step gradient descent),"""
            """self.agg let the server aggregate local models,"""
            """self.broadcast replaces the selected_clients' local models with the aggregated server model."""
            self.update(selected_clients, lr)
            self.agg(selected_clients)
            self.broadcast(selected_clients)
            
            if self.step_decay == 'sqrt':
                lr = lr0 / np.sqrt(round)
            elif self.step_decay == 'linear':
                lr = lr0 / round

            if round % 2 == 0:
                self.test(criterion, round, tb, 
                        n_swap=n_swap,algorithm=algorithm,speedup=self.speedup)
            
            end = datetime.datetime.now()
            print(f"\r| Train | communication_round: {round}|{communication_round}, time: {(end - start).seconds}s",
                flush=True, end="")
    
    def test(self, criterion, epoch,tb, n_swap=None,algorithm='AMFED',speedup=0, TEST_ACCURACY=0):
        model = self.server.global_model
        train_loader = self.temp_train_loader
        test_loader = self.temp_test_loader
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

    def broadcast(self,selected_clients):
        for index in selected_clients:
            self.worker_list[index].load_model_params(self.server.get_model_params())

    @abstractmethod
    def update(self,selected_clients):
        pass

    @abstractmethod
    def agg(self,selected_clients):
        pass

def dropout_init(stat,dropout,threshold,alpha,repetition,interval,**kwargs):
    if dropout == 'forever':
        dropout_class = forever_scheme(threshold,alpha,stat)
    elif dropout == 'alternate':
        dropout_class = alternate_scheme(threshold,alpha,stat,repetition,interval)     
    else:
        print('Dropout scheme does not exist! Try again.')
        sys.exit()
    dropout_class.prepare()
    return dropout_class    
