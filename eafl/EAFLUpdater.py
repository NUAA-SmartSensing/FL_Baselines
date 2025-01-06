import torch.utils.data

from core.handlers.Handler import HandlerChain, Handler, TreeFilter
from core.handlers.ModelTestHandler import ServerPostTestHandler, ServerTestHandler
from updater.SyncUpdater import SyncUpdater
from utils import ModuleFindTool


class EAFLUpdater(SyncUpdater):
    def __init__(self, server_thread_lock, config, mutex_sem, empty_sem, full_sem):
        SyncUpdater.__init__(self, server_thread_lock, config, mutex_sem, empty_sem, full_sem)
        self.group_manager = self.global_var["group_manager"]

        group_update_class = ModuleFindTool.find_class_by_path(config["group"]["path"])
        self.group_update = group_update_class(self.config["group"]["params"])
        self.client_list = self.global_var["client_id_list"]
        self.selected_list = []
        self.r = self.global_var["server_config"]["scheduler"].get("r", 30)

    def create_handler_chain(self):
        self.handler_chain = HandlerChain()
        rFilter = RFliter()
        self.handler_chain.set_chain(rFilter)
        reGroup = EAFLReGroup()
        aggregation = EAFLAggregation()
        rFilter.add_child(reGroup)
        rFilter.add_child(aggregation)
        test = ServerTestHandler()
        test.set_next(ServerPostTestHandler())
        aggregation.set_next(test)
        reGroup.set_next(test)


class RFliter(TreeFilter):
    def _handle(self, request):
        epoch = request.get('epoch')
        scheduler = request.get('scheduler')
        if epoch % scheduler.r == 1:
            return 0
        return 1


class EAFLReGroup(Handler):
    def _handle(self, request):
        updater = request.get('updater')
        # 等待所有人上传梯度
        print("分组")
        update_list = []
        for i in range(updater.group_manager.get_group_num()):
            for _ in range(len(updater.group_manager.get_group_list()[i])):
                update_list.append(updater.queue_manager.get(i))
        # 进行分组
        group_list, _ = updater.group_manager.update(update_list)
        updater.global_var['scheduler'].send_group_info(group_list)
        updater.selected_list = updater.client_list
        updater.global_var['scheduler'].set_selected_clients(updater.client_list)
        return request


class EAFLAggregation(Handler):
    def _handle(self, request):
        # 记录一下客户端id
        updater = request.get('updater')
        epoch = request.get('epoch')
        group_list = updater.group_manager.get_group_list()
        total_data_list = [0 for i in range(len(group_list))]
        # 每个组最近的k个客户端的更新
        selected_list = []
        for i in range(updater.group_manager.get_group_num()):
            update_list = []
            while len(update_list) < 0.5 * len(group_list[i]):
                update_list.append(updater.queue_manager.get(i))
                selected_list.append(update_list[len(update_list)-1]['client_id'])
                total_data_list[i] += update_list[len(update_list)-1]['data_sum']
            # 组内更新
            updater.group_manager.network_list[i] = self.inner_group_aggregation(updater, epoch, update_list)
        # 组间更新
        self.inter_group_aggregation(updater, epoch, updater.group_manager.network_list, total_data_list)
        return request

    @staticmethod
    def inner_group_aggregation(updater, epoch, update_list):
        global_model, _ = updater.update_caller.update_server_weights(epoch, update_list)
        if torch.cuda.is_available():
            for key, var in global_model.items():
                global_model[key] = global_model[key].cuda()
        return global_model

    @staticmethod
    def inter_group_aggregation(updater, epoch, network_list, data_list):
        update_list = []
        for i in range(updater.global_var['group_manager'].group_num):
            update_list.append({"weights": network_list[i], "data_sum": data_list[i]})
        global_model, delivery_weights = updater.group_update.update_server_weights(epoch, update_list)
        updater.global_var['delivery_weights'] = delivery_weights
        updater.model.load_state_dict(global_model)
