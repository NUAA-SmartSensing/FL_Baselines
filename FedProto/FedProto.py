import copy

import torch
from torch import nn
from torch.utils import model_zoo
from torch.utils.data import DataLoader

from client.NormalClient import NormalClient
from client.mixin.ClientHandler import UpdateReceiver
from core.handlers.Handler import Handler, HandlerChain
from core.handlers.ModelTestHandler import ClientTestHandler
from core.handlers.ServerHandler import ClientUpdateGetter, Aggregation
from lib.FedProto.models.models import CNNMnist, CNNFemnist
from lib.FedProto.models.resnet import resnet18, model_urls
from update.AbstractUpdate import AbstractUpdate
from updater.SyncUpdater import SyncUpdater
from utils import ModuleFindTool
from utils.DatasetUtils import FLDataset


def _get_transform(config):
    transform, target_transform = None, None
    if "transform" in config:
        transform_func = ModuleFindTool.find_class_by_path(config["transform"]["path"])
        transform = transform_func(**config["transform"]["params"])
    if "target_transform" in config:
        target_transform_func = ModuleFindTool.find_class_by_path(config["target_transform"]["path"])
        target_transform = target_transform_func(**config["target_transform"]["params"])
    return transform, target_transform


class ProtoClient(NormalClient):
    def __init__(self, c_id, stop_event, selected_event, delay, index_list, config, dev):
        super().__init__(c_id, stop_event, selected_event, delay, index_list, config, dev)
        self.global_protos = None
        self.ld = config.get("ld", 1)

    def init(self):
        super().init()
        self.test_ds = self.message_queue.get_test_dataset()
        transform, target_transform = _get_transform(self.config)
        self.fl_test_ds = FLDataset(self.test_ds, list(range(len(self.test_ds))), transform, target_transform)
        self.test_dl = DataLoader(self.fl_test_ds, batch_size=self.batch_size, shuffle=True, drop_last=True)
        self.classes_list = set(self.fl_train_ds.dataset.targets[self.index_list].numpy())

    def create_handler_chain(self):
        super().create_handler_chain()
        self.handler_chain.exchange_handler(ProtoReceiver(), UpdateReceiver)
        self.handler_chain.add_handler(ClientTestHandler())

    def train(self):
        for iter in range(self.epoch):
            agg_protos_label = {}
            for idx, (data, label_g) in enumerate(self.train_dl):
                self.optimizer.zero_grad()
                data, labels = data.to(self.dev), label_g.to(self.dev)
                preds, protos = self.model(data)
                loss1 = self.loss_func(preds, labels)
                loss_mse = nn.MSELoss()
                if len(self.global_protos) == 0:
                    loss2 = 0*loss1
                else:
                    proto_new = copy.deepcopy(protos.data)
                    i = 0
                    for label in labels:
                        if label.item() in self.global_protos.keys():
                            proto_new[i, :] = self.global_protos[label.item()][0].data
                        i += 1
                    loss2 = loss_mse(proto_new, protos)
                loss = loss1 + loss2 * self.ld
                loss.backward()
                self.optimizer.step()

                for i in range(len(labels)):
                    if label_g[i].item() in agg_protos_label:
                        agg_protos_label[label_g[i].item()].append(protos[i,:])
                    else:
                        agg_protos_label[label_g[i].item()] = [protos[i,:]]
        return 0, agg_func(agg_protos_label)

    def test(self):
        """ Returns the test accuracy and loss.
        """
        loss, total, correct = 0.0, 0.0, 0.0
        loss_mse = nn.MSELoss()

        criterion = nn.NLLLoss().to(self.dev)

        acc_list_g = []
        acc_list_l = []
        loss_list = []
        # test (local model)
        self.model.eval()
        for batch_idx, (images, labels) in enumerate(self.test_dl):
            images, labels = images.to(self.dev), labels.to(self.dev)
            self.model.zero_grad()
            outputs, protos = self.model(images)

            batch_loss = criterion(outputs, labels)
            loss += batch_loss.item()

            # prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        acc = correct / total
        print('| User: {} | Global Test Acc w/o protos: {:.3f}'.format(self.client_id, acc))
        if self.global_protos != []:
            for batch_idx, (images, labels) in enumerate(self.test_dl):
                images, labels = images.to(self.dev), labels.to(self.dev)
                self.model.zero_grad()
                outputs, protos = self.model(images)
                num_classes = self.config['model']['params']['num_classes']
                # compute the dist between protos and global_protos
                a_large_num = 100
                dist = a_large_num * torch.ones(size=(images.shape[0], num_classes)).to(self.dev)  # initialize a distance matrix
                for i in range(images.shape[0]):
                    for j in range(num_classes):
                        if j in self.global_protos.keys() and j in self.classes_list:
                            d = loss_mse(protos[i, :], self.global_protos[j][0])
                            dist[i, j] = d

                # prediction
                _, pred_labels = torch.min(dist, 1)
                pred_labels = pred_labels.view(-1)
                correct += torch.sum(torch.eq(pred_labels, labels)).item()
                total += len(labels)

                # compute loss
                proto_new = copy.deepcopy(protos.data)
                i = 0
                for label in labels:
                    if label.item() in self.global_protos.keys():
                        proto_new[i, :] = self.global_protos[label.item()][0].data
                    i += 1
                loss2 = loss_mse(proto_new, protos)
                loss2 = loss2.cpu().detach().numpy()

            acc = correct / total
            print('| User: {} | Global Test Acc with protos: {:.5f}'.format(self.client_id, acc))
            acc_list_g.append(acc)
            loss_list.append(loss2)
        return None, None


def agg_func(protos):
    """
    Returns the average of the weights.
    """

    for [label, proto_list] in protos.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            protos[label] = proto / len(proto_list)
        else:
            protos[label] = proto_list[0]

    return protos


def proto_aggregation(local_protos_list):
    agg_protos_label = dict()
    for idx in local_protos_list:
        local_protos = local_protos_list[idx]
        for label in local_protos.keys():
            if label in agg_protos_label:
                agg_protos_label[label].append(local_protos[label])
            else:
                agg_protos_label[label] = [local_protos[label]]

    for [label, proto_list] in agg_protos_label.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            agg_protos_label[label] = [proto / len(proto_list)]
        else:
            agg_protos_label[label] = [proto_list[0].data]

    return agg_protos_label


class ProtoReceiver(Handler):
    def _handle(self, request):
        client = request.get('client')
        weights_buffer = client.message_queue.get_from_downlink(client.client_id, 'weights')
        client.global_protos = weights_buffer
        client.time_stamp = client.message_queue.get_from_downlink(client.client_id, 'time_stamp')
        client.schedule_t = client.message_queue.get_from_downlink(client.client_id, 'schedule_time_stamp')
        client.receive_notify()
        return request


class FedProto(AbstractUpdate):
    def __init__(self, config):
        self.config = config

    def update_server_weights(self, epoch, update_list):
        protos = {update["client_id"]: update['weights'] for update in update_list}
        return proto_aggregation(protos), None


class ProtoUpdater(SyncUpdater):
    def create_handler_chain(self):
        self.handler_chain = HandlerChain()
        (self.handler_chain.set_chain(ClientUpdateGetter())
         .set_next(Aggregation()))


def create_proto_model(src_obj, **args):
    if args["dataset"] == 'mnist':
        if args["mode"] == 'model_heter':
            if src_obj.client_id < 7:
                args["out_channels"] = 18
            elif src_obj.client_id >= 7 and src_obj.client_id < 14:
                args["out_channels"] = 20
            else:
                args["out_channels"] = 22
        else:
            args["out_channels"] = 20

        local_model = CNNMnist(args=args)

    elif args["dataset"] == 'femnist':
        if args["mode"] == 'model_heter':
            if src_obj.client_id < 7:
                args["out_channels"] = 18
            elif src_obj.client_id >= 7 and src_obj.client_id < 14:
                args["out_channels"] = 20
            else:
                args["out_channels"] = 22
        else:
            args["out_channels"] = 20
        local_model = CNNFemnist(args=args)

    elif args["dataset"] == 'cifar100' or args["dataset"] == 'cifar10':
        if args["mode"] == 'model_heter':
            if src_obj.client_id < 10:
                args["stride"] = [1, 4]
            else:
                args["stride"] = [2, 2]
        else:
            args["stride"] = [2, 2]
        resnet = resnet18(args, pretrained=False, num_classes=args["num_classes"])
        initial_weight = model_zoo.load_url(model_urls['resnet18'])
        local_model = resnet
        initial_weight_1 = local_model.state_dict()
        for key in initial_weight.keys():
            if key[0:3] == 'fc.' or key[0:5] == 'conv1' or key[0:3] == 'bn1':
                initial_weight[key] = initial_weight_1[key]

        local_model.load_state_dict(initial_weight)
    else:
        raise ValueError("Unknown dataset: {}".format(args["dataset"]))
    return local_model
