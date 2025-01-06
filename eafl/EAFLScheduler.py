from scheduler.SyncScheduler import SyncScheduler


class EAFLScheduler(SyncScheduler):
    def __init__(self, server_thread_lock, config, mutex_sem, empty_sem, full_sem):
        super().__init__(server_thread_lock, config, mutex_sem, empty_sem, full_sem)
        self.r = config.get('r', 30)
        self.group_manager = self.global_var['group_manager']
        self.selected_clients = self.global_var['client_id_list']
        self.message_queue.create_downlink("changed_group")

    def client_select(self):
        return self.selected_clients

    def wait(self):
        current_time = self.current_t.get_time()
        # 等待所有客户端上传更新
        group_list = self.group_manager.get_group_list()
        if current_time % self.r == 1:
            l = [len(i) for i in group_list]
        else:
            l = [0.5 * len(i) for i in group_list]
        self.queue_manager.receive(l)

    def set_selected_clients(self, selected_clients):
        self.selected_clients = selected_clients

    def send_group_info(self, group_list):
        for i in range(len(group_list)):
            for j in group_list[i]:
                self.message_queue.put_into_downlink(j, 'group_id', i)
                self.message_queue.put_into_downlink(j, 'changed_group', True)
