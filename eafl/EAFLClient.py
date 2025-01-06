from client.NormalClient import NormalClient


class EAFLClient(NormalClient):
    def __init__(self, c_id, stop_event, selected_event, delay, index_list, config, dev):
        NormalClient.__init__(self, c_id, stop_event, selected_event, delay, index_list, config, dev)
        self.group_id = 0

    def customize_upload(self):
        self.upload_item("group_id", self.group_id)

    def receive_notify(self):
        self.message_queue.put_into_downlink(self.client_id, 'changed_group', False)
        self.group_id = self.message_queue.get_from_downlink(self.client_id, 'group_id')
        if self.group_id is None:
            self.group_id = 0
