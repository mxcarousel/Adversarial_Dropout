import torch
from .__init__ import trainer

class mifa_trainer(trainer):    
    def update(self,selected_clients, worker_list, server, lr):
        for index in selected_clients:
            worker = worker_list[index]
            """Now do $s$ steps local training at the chosen client"""
            for step in range(self.local_steps):
                try:
                    worker.step()
                except StopIteration:
                    worker.update_iter()
                    worker.step()
                if self.step_decay:
                    worker.update_grad_decay(lr)
                else:
                    worker.update_grad()

    def agg(self,selected_clients, worker_list, server):
        selected_clients = selected_clients.reshape(-1)
        weight_tensor = torch.tensor(server.weight)
        index = 0
        for index in selected_clients:
            for name, param in worker_list[index].G1.named_parameters():
                param.data = (server.get_model_params()[name].data -worker_list[index].get_model_params()[name].data)/worker_list[index].get_lr()
            for name, param in server.G.named_parameters():
                param.data += (worker_list[index].get_G1_params()[name].data - 
                                worker_list[index].get_G0_params()[name].data) * weight_tensor[index]
            worker_list[index].load_G0_params(worker_list[index].get_G1_params())
        for name, param in server.global_model.named_parameters():
            param.data -= server.G.state_dict()[name].data * server[index].get_lr()
    