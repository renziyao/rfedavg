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
    
    def calculate_loss(self):
        loss_meter = AvgMeter()
        with torch.no_grad():
            for i, data in enumerate(self.trainloader):
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(inputs)
                classifier_loss = self.classifier_criterion(
                    outputs,
                    labels,
                )
                loss_meter.append(classifier_loss.item())
        return loss_meter.avg()
    
    def local_train(self):
        omega = self.model.parameters_to_tensor().clone().detach()
        omega_loss = self.calculate_loss()
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
                classifier_loss.backward()
                self.optimizer.step()
                self.meters['classifier_loss'].append(classifier_loss.item())
                batch_count += 1
        with torch.no_grad():
            L = self.params['Trainer']['L']
            q = self.params['Trainer']['q']
            delta_omega = L * (omega - self.model.parameters_to_tensor())
            self.delta = delta_omega * (omega_loss ** q)
            self.h = q * (omega_loss ** (q - 1)) * (torch.norm(self.delta) ** 2) + L * (omega_loss ** q)

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
    
    def aggregate_model(self, clients):
        n = len(clients)
        with torch.no_grad():
            omega_old = self.center.model.parameters_to_tensor()
            numerator = torch.zeros_like(omega_old)
            denominator = 0.0
            for client in clients:
                numerator += client.delta
                denominator += client.h
            omega = omega_old - numerator / denominator
        self.center.model.tensor_to_parameters(omega)
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
                for p in client.optimizer.param_groups:
                    p['lr'] = self.learning_rate
            
            for client in clients:
                # local train
                client.local_train()
            
            # aggregate params
            self.aggregate_model(clients)

            self.learning_rate *= self.params['Trainer']['optimizer']['lr_decay']

            time_end = time.time()

            if round % self.test_interval == 0:
                print('Summary, Accuracy: %.5f, Time: %.0fs' % (
                    self.center.test_accuracy(),
                    time_end - time_begin,
                ))
            print('%sRound %d end%s' % ('=' * 10, round, '=' * 10))
