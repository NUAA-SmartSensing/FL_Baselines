from collections import OrderedDict

import torch
from torch.utils.data import DataLoader

from client.NormalClient import NormalClient
from update.AbstractUpdate import AbstractUpdate
from utils.Algorithm import bernoulli_sampling
from utils.GlobalVarGetter import GlobalVarGetter


class SampleClient(NormalClient):
    def __init__(self, c_id, stop_event, selected_event, delay, index_list, config, dev):
        NormalClient.__init__(self, c_id, stop_event, selected_event, delay, index_list, config, dev)
        self.sample_p = config.get("sample_p", 0.1)

    def train(self):
        self.model.train()
        index_list = bernoulli_sampling(self.index_list, self.sample_p)
        self.fl_train_ds.change_idxs(index_list)
        if len(index_list) == 0:
            return 0, {k: v.grad.clone().detach() for k, v in self.model.named_parameters()}
        self.train_dl = DataLoader(self.fl_train_ds, batch_size=len(index_list), shuffle=True)
        data_sum = 0
        for data, label in self.train_dl:
            self.optimizer.zero_grad()
            data, label = data.to(self.dev), label.to(self.dev)
            preds = self.model(data)
            loss = self.loss_func(preds, label)
            data_sum += label.size(0)
            loss.backward()
            break
        gradients = OrderedDict()
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                gradients[name] = param.grad.clone().detach() * len(index_list)
        return data_sum, gradients


class SampleUpdate(AbstractUpdate):
    def __init__(self, config):
        self.global_var = GlobalVarGetter.get()
        self.lr = config.get("lr", 0.01)

    def update_server_weights(self, epoch, update_list):
        global_model = self.global_var["global_model"].state_dict()
        total_nums = 0
        for update_dict in update_list:
            total_nums += update_dict["data_sum"]
        updated_parameters = {k: torch.zeros_like(v) for k, v in global_model.items()}
        for update_dict in update_list:
            if update_dict["data_sum"] == 0:
                continue
            client_weights = update_dict["weights"]
            for key, var in client_weights.items():
                updated_parameters[key] += client_weights[key] / total_nums
        # print(updated_parameters['conv1.weight'][0][0][0], total_nums)
        for key, var in global_model.items():
            updated_parameters[key] = var - self.lr * updated_parameters[key]
        return updated_parameters, None
