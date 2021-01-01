from src.trainers.base import BaseClient, BaseServer, AvgMeter

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import time
import copy
from src.utils import *


class Client(BaseClient):
    def __init__(self, id, params, trainset, testset):
        super().__init__(id, params, trainset, testset)
        self.classifier_criterion = nn.CrossEntropyLoss()
        self.mmd_criterion = LinearMMD()
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
            'omega_loss': AvgMeter(),
        }
    
    def local_train(self):
        self.set_omega()
        batch_count = 0
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
                omega_loss = 0.0
                for (name, params) in self.model.named_parameters():
                    omega_loss += torch.norm(params.data - self.omega[name]) ** 2
                loss = classifier_loss + omega_loss * self.params['Trainer']['lambda']
                loss.backward()
                self.optimizer.step()
                self.meters['classifier_loss'].append(classifier_loss.item())
                self.meters['omega_loss'].append(omega_loss.item())
                batch_count += 1
        print('Client %d, classifier_loss: %.5f, omega_loss: %.5f, acc: %.5f' % (
            self.id, 
            self.meters['classifier_loss'].avg(-batch_count),
            self.meters['omega_loss'].avg(-batch_count),
            self.test_accuracy(),
        ), flush=True)
        self.optimizer.param_groups[0]['lr'] *= self.params['Trainer']['optimizer']['lr_decay']
        self.params['Trainer']['lambda'] *= self.params['Trainer']['lambda_decay']

    def set_omega(self):
        self.omega = copy.deepcopy(self.model.state_dict())
        return


class Server(BaseServer):
    def __init__(self, params):
        super().__init__(params)
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

    def train(self):
        for round in range(1, self.T + 1):
            print('%sRound %d begin%s' % ('=' * 10, round, '=' * 10))

            time_begin = time.time()
            # random clients
            clients = self.sample_client()

            # for each client in choose_clients
            for client in clients:
                # send params
                client.clone_model(self.center)
                
                # local train
                client.local_train()
            
            # aggregate params
            self.center.aggregate_model(self.clients)

            time_end = time.time()

            if round % self.TEST_INTERVAL == 0:
                print('Summary, Accuracy: %.5f, Time: %.0fs' % (
                    self.center.test_accuracy(),
                    time_end - time_begin,
                ))
            print('%sRound %d end%s' % ('=' * 10, round, '=' * 10))
