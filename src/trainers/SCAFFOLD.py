from os import access
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import time
from src.trainers.base import *


class Client(BaseClient):
    def __init__(self, id, params, trainset, testset):
        super().__init__(id, params, trainset, testset)
        self.classifier_criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(
            [
                {'params': self.model.parameters()},
            ], 
            lr=params['Trainer']['optimizer']['lr'],
            momentum=params['Trainer']['optimizer']['momentum'],
            weight_decay=params['Trainer']['optimizer']['weight_decay'],
        )
        self.params = params
        self.meters = {
            'classifier_loss': AvgMeter(), 
        }
        self.c = None
        self.c_i = torch.zeros_like(self.model.parameters_to_tensor())
        self.c_i_plus = None
        self.delta_c = None
        self.delta_y = None
    
    def local_train(self):
        batch_count = 0
        x = self.model.parameters_to_tensor().detach().clone()
        eta = self.optimizer.param_groups[0]['lr']
        for epoch in range(self.E):
            for i, data in enumerate(self.trainloader):
                self.optimizer.zero_grad()
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(inputs)
                classifier_loss = self.classifier_criterion(
                    outputs,
                    labels,
                )
                classifier_loss.backward()
                self.optimizer.step()
                with torch.no_grad():
                    p_tensor = self.model.parameters_to_tensor()
                    p_tensor.add_(eta * (self.c_i - self.c))
                self.meters['classifier_loss'].append(classifier_loss.item())
                batch_count += 1
        y = self.model.parameters_to_tensor()
        with torch.no_grad():
            self.c_i_plus = self.c_i - self.c + 1 / (batch_count * eta) * (x - y)
            self.delta_y = y - x
            self.delta_c = self.c_i_plus - self.c_i
            self.c_i = self.c_i_plus
            del self.c
            del self.c_i_plus
        print('Client %d, classifier_loss: %.5f, acc: %.5f' % (
            self.id, 
            self.meters['classifier_loss'].avg(-batch_count),
            self.test_accuracy(),
        ), flush=True)


class Server(BaseServer):
    def __init__(self, params):
        super().__init__(params)
        self.learning_rate = self.params['Trainer']['optimizer']['lr']
        self.center = Client(0, params, None, self.testset)
        self.clients = []
        for i in range(self.n_clients):
            self.clients.append(
                Client(
                    i + 1, 
                    params,
                    self.dataset_split[i]['train'],
                    self.dataset_split[i]['test'],
                )
            )
        self.c = torch.zeros_like(self.center.model.parameters_to_tensor())

    def aggregate_model(self, clients):
        n = len(clients)
        N = len(self.clients)
        eta_g = self.params['Trainer']['eta_g']
        with torch.no_grad():
            delta_x = []
            delta_c = []
            for _, client in enumerate(clients):
                delta_x.append(client.delta_y)
                delta_c.append(client.delta_c)
            delta_x = 1 / n * sum(delta_x)
            delta_c = 1 / n * sum(delta_c)
            self.center.model.tensor_to_parameters(
                self.center.model.parameters_to_tensor() + (delta_x * eta_g)
            )
            self.c.add_(n / N * delta_c)
            for _, client in enumerate(clients):
                del client.delta_y
                del client.delta_c
        return

    def train(self):
        for round in range(1, self.Round + 1):
            print('%sRound %d begin%s' % ('=' * 10, round, '=' * 10))

            time_begin = time.time()
            # random clients
            clients = self.sample_client()
            for client in clients:
                # send params
                client.clone_model(self.center)
                client.c = self.c
                for p in client.optimizer.param_groups:
                    p['lr'] = self.learning_rate

            # for each client in choose_clients
            for client in clients:
                # local train
                client.local_train()
            
            # aggregate params
            self.aggregate_model(clients)

            self.learning_rate *= self.params['Trainer']['optimizer']['lr_decay']

            time_end = time.time()

            if round % self.test_interval == 0:
                acc = self.center.test_accuracy()
                print('Summary, Accuracy: %.5f, Time: %.0fs' % (
                    acc,
                    time_end - time_begin,
                ))
                self.acc_meter.append(acc)
            print('%sRound %d end%s' % ('=' * 10, round, '=' * 10))
        print('Done, max acc: %.5f' % (self.acc_meter.max()))
