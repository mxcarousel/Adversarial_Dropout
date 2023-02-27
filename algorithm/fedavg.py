from .__init__ import trainer
import torch

class fedavg_trainer(trainer):
    def update(self,selected_clients, lr):
        for index in selected_clients:
            worker = self.worker_list[index]
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

    def agg(self,selected_clients):
        selected_clients = selected_clients.reshape(-1)
        weight_tensor = torch.tensor(self.server.weight)
        selected_weight = weight_tensor[selected_clients].sum()
        weight_tensor /= selected_weight # renormalization
        for name, param in self.server.global_model.named_parameters():
            param.data *= 0
            for index in selected_clients:
                param.data += self.worker_list[index].get_model_params()[name].data * weight_tensor[index]
    