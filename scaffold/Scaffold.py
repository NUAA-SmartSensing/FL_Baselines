import torch

from update.AbstractUpdate import AbstractUpdate
from utils.GlobalVarGetter import GlobalVarGetter
from utils.Tools import to_dev


class Scaffold(AbstractUpdate):
    def __init__(self, config):
        self.config = config
        self.lr = config["lr"] if "lr" in config else 1.0
        self.dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.global_var = GlobalVarGetter.get()
        self.control = {}
        self.global_weights = self.global_var['server_network'].state_dict()
        for k in self.global_var['server_network'].state_dict().keys():
            self.control[k] = to_dev(torch.zeros_like(self.global_weights[k].cpu()), self.dev)
        self.global_var['control'] = self.control

    def update_server_weights(self, epoch, update_list):
        client_num = len(update_list)
        updated_parameters = {}
        new_control = {}
        for key in update_list[0]["weights"].keys():
            updated_parameters[key] = to_dev(torch.zeros_like(update_list[0]["weights"][key], dtype=torch.float32), self.dev)
        for key in update_list[0]["delta_c"].keys():
            new_control[key] = to_dev(torch.zeros_like(update_list[0]["delta_c"][key], dtype=torch.float32), self.dev)
        for update in update_list:
            for key in update["weights"].keys():
                updated_parameters[key] += update["weights"][key] / client_num
            for key in update["delta_c"].keys():
                new_control[key] += update["delta_c"][key] / client_num
        for key in new_control.keys():
            self.control[key] += client_num * new_control[key] / self.global_var["global_config"]["client_num"]
        for key in updated_parameters.keys():
            updated_parameters[key] = self.global_weights[key] + self.lr * updated_parameters[key]
        return updated_parameters, None
