import time

import torch.utils.data

from updater.SyncUpdater import SyncUpdater
from utils import ModuleFindTool


class EAFLUpdater(SyncUpdater):
    def __init__(self, server_thread_lock, stop_event, config, mutex_sem, empty_sem, full_sem):
        SyncUpdater.__init__(self, server_thread_lock, stop_event, config, mutex_sem, empty_sem, full_sem)
        self.group_manager = self.global_var["group_manager"]

        group_update_class = ModuleFindTool.find_class_by_path(config["group"]["path"])
        self.group_update = group_update_class(self.config["group"]["params"])
        self.client_list = self.global_var["client_id_list"]
        self.selected_list = []
        self.r = self.global_var["server_config"]["scheduler"]["r"] if "r" in self.global_var["server_config"]["scheduler"] else 30

    def run(self):
        for _ in range(self.T):
            self.full_sem.acquire()
            self.mutex_sem.acquire()

            epoch = self.current_time.get_time()
            update_list = []
            if epoch % self.r == 1:
                # 等待所有人上传梯度
                print("分组")
                for i in range(self.group_manager.get_group_num()):
                    for _ in range(len(self.group_manager.get_group_list()[i])):
                        update_list.append(self.queue_manager.get(i))
                # 进行分组
                group_list, _ = self.group_manager.update(update_list)
                self.global_var['scheduler'].send_group_info(group_list)
                self.selected_list = self.client_list
            else:
                # 记录一下客户端id
                group_list = self.group_manager.get_group_list()
                total_data_list = [0 for i in range(len(group_list))]
                # 每个组最近的k个客户端的更新
                self.selected_list = []
                for i in range(self.group_manager.get_group_num()):
                    update_list = []
                    while len(update_list) < 0.5 * len(group_list[i]):
                        update_list.append(self.queue_manager.get(i))
                        self.selected_list.append(update_list[len(update_list)-1]['client_id'])
                        total_data_list[i] += update_list[len(update_list)-1]['data_sum']
                    # 组内更新
                    self.group_manager.network_list[i] = self.update_group_weights(epoch, update_list)
                # 组间更新
                self.update_server_weights(epoch, self.group_manager.network_list, total_data_list)

            self.server_thread_lock.acquire()
            self.run_server_test(epoch)
            self.global_var['scheduler'].set_selected_clients(self.selected_list)
            self.server_thread_lock.release()

            self.current_time.time_add()
            self.mutex_sem.release()
            self.empty_sem.release()
        print("Average delay =", (self.sum_delay / self.T))

    def update_group_weights(self, epoch, update_list):
        global_model, _ = self.group_update.update_server_weights(epoch, update_list)
        if torch.cuda.is_available():
            for key, var in global_model.items():
                global_model[key] = global_model[key].cuda()
        return global_model

    def update_server_weights(self, epoch, network_list, data_list):
        update_list = []
        for i in range(self.global_var['group_manager'].group_num):
            update_list.append({"weights": network_list[i], "data_sum": data_list[i]})
        super().update_server_weights(epoch, update_list)
