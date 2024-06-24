from scheduler.SyncScheduler import SyncScheduler


class ScaffoldScheduler(SyncScheduler):
    def send_weights(self, client_id, current_time, schedule_time):
        self.message_queue.put_into_downlink(client_id, 'control', self.global_var['control'])
        super().send_weights(client_id, current_time, schedule_time)
