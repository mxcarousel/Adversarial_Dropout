from .__init__ import trainer
import torch
import numpy as np

class amfed_prox_trainer(trainer):     
    def update(self,selected_clients, lr):
        for index in selected_clients:
            worker = self.worker_list[index]
            for step in range(self.local_steps):
                try:
                    worker.fedprox_step(self.server.global_model, self.mu)
                except StopIteration:
                    worker.update_iter()
                    worker.fedprox_step(self.server.global_model, self.mu)
                if self.step_decay:
                    worker.update_grad_decay(lr)
                else:
                    worker.update_grad()

    def agg(self,selected_clients):
        selected_clients = selected_clients.reshape(-1)
        weight_tensor = torch.tensor(self.server.weight)
        selected_weight = weight_tensor[selected_clients].sum()
        if self.speedup:
            su = 1/np.sqrt(self.server.prob)
            for name, param in self.server.global_model.named_parameters():
                param.data *= (1-selected_weight)+selected_weight*(1-su)
                for index in selected_clients:
                    param.data += su*self.worker_list[index].get_model_params()[name].data * self.server.weight[index]
        else:
            for name, param in self.server.global_model.named_parameters():
                param.data *= (1-selected_weight)
                for index in selected_clients:
                    param.data += self.worker_list[index].get_model_params()[name].data * self.server.weight[index]