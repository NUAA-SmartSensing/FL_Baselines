import copy

from update.AbstractUpdate import AbstractUpdate
from utils.GlobalVarGetter import GlobalVarGetter
from utils.Tools import to_dev


class SAA(AbstractUpdate):
    def __init__(self, config):
        self.config = config
        self.global_var = GlobalVarGetter().get()

    def update_server_weights(self, epoch, update_list):
        server_weights = to_dev(copy.deepcopy(self.global_var['updater'].model.state_dict()), 'cuda')
        total_nums = 0
        for update_dict in update_list:
            total_nums += update_dict["data_sum"]
            update_dict["stale"] = 1 / (epoch - update_dict["time_stamp"] + 1)
        updated_parameters = {}
        if total_nums:
            for key, var in update_list[0]["weights"].items():
                updated_parameters[key] = update_dict["stale"] * update_list[0]["data_sum"] * (
                            update_list[0]["weights"][key] - server_weights[key]) / total_nums
            for i in range(len(update_list) - 1):
                update_dict = update_list[i + 1]
                client_weights = update_dict["weights"]
                for key, var in client_weights.items():
                    updated_parameters[key] += update_dict["stale"] * (client_weights[key] - server_weights[key]) * \
                                               update_dict["data_sum"] / total_nums
            for k, _ in updated_parameters.items():
                updated_parameters[k] += server_weights[k]
        else:
            updated_parameters = server_weights
        return updated_parameters, None


class DSA(AbstractUpdate):
    def __init__(self, config):
        self.config = config
        self.global_var = GlobalVarGetter().get()

    def update_server_weights(self, epoch, update_list):
        total_nums = 0
        for update_dict in update_list:
            total_nums += update_dict["data_sum"]
        updated_parameters = {}
        for key, var in update_list[0]["weights"].items():
            updated_parameters[key] = update_list[0]["weights"][key] * update_list[0]["data_sum"] / total_nums
        for i in range(len(update_list) - 1):
            update_dict = update_list[i + 1]
            client_weights = update_dict["weights"]
            for key, var in client_weights.items():
                updated_parameters[key] += client_weights[key] * update_dict["data_sum"] / total_nums
        return updated_parameters, None
