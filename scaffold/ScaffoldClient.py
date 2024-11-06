import copy

import torch

from client.NormalClient import NormalClientWithDelta
from utils.GlobalVarGetter import GlobalVarGetter
from utils.Tools import to_cpu, to_dev


class ScaffoldClient(NormalClientWithDelta):
    def __init__(self, c_id, stop_event, selected_event, delay, index_list, config, dev):
        super().__init__(c_id, stop_event, selected_event, delay, index_list, config, dev)
        self.control = {}
        self.global_control = {}
        self.delta_c = {}

    def train_one_epoch(self):
        x = copy.deepcopy(self.model)
        data_sum = 0
        for _ in range(self.epoch):
            for data, label in self.train_dl:
                data, label = data.to(self.dev), label.to(self.dev)
                preds = self.model(data)
                # Calculate the loss function
                loss = self.loss_func(preds, label)
                data_sum += label.size(0)
                # proximal term
                if self.mu != 0:
                    proximal_term = 0.0
                    for w, w_t in zip(self.model.parameters(), x.parameters()):
                        proximal_term += (w - w_t).norm(2)
                    loss = loss + (self.mu / 2) * proximal_term
                # backpropagate
                loss.backward()
                for k, v in self.model.named_parameters():
                    v.grad += self.global_control[k].data - self.control[k].data
                # Update the gradient
                self.opti.step()
                if self.lr_scheduler:
                    self.lr_scheduler.step()
                # Zero out the gradient and initialize the gradient.
                self.opti.zero_grad()
        # Return the updated model parameters obtained by training on the client's own data.
        weights = self.model.state_dict()
        # update c
        temp = {}
        for k, v in self.model.named_parameters():
            temp[k] = v.data.clone()

        for k, v in x.named_parameters():
            local_steps = self.epoch * len(self.train_dl)
            raw = copy.deepcopy(self.control[k])
            self.control[k] = self.control[k] - self.global_control[k] + (v.data - temp[k]) / (local_steps * self.opti.state_dict()['param_groups'][0]['lr'])
            self.delta_c[k] = self.control[k] - raw
            weights[k] = temp[k] - v.data
        return data_sum, weights

    def customize_upload(self):
        self.upload_item("delta_c", to_cpu(self.delta_c))

    def init_client(self):
        super().init_client()
        for k, v in self.model.named_parameters():
            self.control[k] = torch.zeros_like(v)
            self.global_control[k] = torch.zeros_like(v)
        self.control = to_dev(self.control, self.dev)
        self.global_control = to_dev(self.global_control, self.dev)
        global_var = GlobalVarGetter.get()
        global_var['server_controls'] = self.global_control
        global_var['client_controls'] = self.control

    def receive_notify(self):
        super().receive_notify()
        self.global_control = to_dev(copy.deepcopy(self.message_queue.get_from_downlink(self.client_id, "control")), self.dev)
