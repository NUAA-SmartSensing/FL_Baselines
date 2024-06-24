import copy

import torch

from client.NormalClient import NormalClientWithDelta
from utils.Tools import to_dev, to_cpu


class ScaffoldClient(NormalClientWithDelta):
    def __init__(self, c_id, stop_event, selected_event, delay, index_list, config, dev):
        super().__init__(c_id, stop_event, selected_event, delay, index_list, config, dev)
        self.control = {}
        self.global_control = None
        self.delta_c = {}

    def train_one_epoch(self):
        x = copy.deepcopy(self.model)
        data_sum, weights = super().train_one_epoch()
        # update c
        self.control = to_dev(self.control, self.dev)
        self.global_control = to_dev(self.global_control, self.dev)
        temp = {}
        for k, v in self.model.named_parameters():
            temp[k] = v.data.clone()

        for k, v in x.named_parameters():
            raw = copy.deepcopy(self.control[k])
            local_steps = self.epoch * len(self.train_dl)
            self.control[k] = self.control[k] - self.global_control[k] + (v.data - temp[k]) / (local_steps * self.opti.state_dict()['param_groups'][0]['lr'])
            self.delta_c[k] = self.control[k] - raw
        return data_sum, weights

    def upload(self, data_sum, weights):
        update_dict = {"client_id": self.client_id, "weights": weights, "data_sum": data_sum,
                       "time_stamp": self.time_stamp, "delta_c": to_cpu(self.delta_c)}
        self.message_queue.put_into_uplink(update_dict)
        print("Client", self.client_id, "uploaded")

    def init_client(self):
        super().init_client()
        for k, v in self.model.state_dict().items():
            self.control[k] = torch.zeros_like(v.cpu())

    def receive_notify(self):
        super().receive_notify()
        self.global_control = self.message_queue.get_from_downlink(self.client_id, "control")
