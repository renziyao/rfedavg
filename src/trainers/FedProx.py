import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import time
from src.trainers.base import *


class Client(BaseClient):
    def __init__(self, id, params, dataset):
        super().__init__(id, params, dataset)
        self.classifier_criterion = nn.CrossEntropyLoss()
        self.optimizer = eval('optim.%s' % params['Trainer']['optimizer']['name'])(
            self.model.parameters(), 
            **params['Trainer']['optimizer']['params'],
        )
        self.params = params
        self.meters = {
            'classifier_loss': AvgMeter(), 
            'omega_loss': AvgMeter(),
        }
    
    def local_train(self):
        self.omega = self.model.parameters_to_tensor().clone().detach()
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
                omega_loss = torch.norm(
                    self.model.parameters_to_tensor() - self.omega
                ) ** 2
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


class Server(BaseServer):
    def __init__(self, params):
        self.Client = Client
        super().__init__(params)
    
    def aggregate_model(self, clients):
        n = len(clients)
        p_tensors = []
        for _, client in enumerate(clients):
            p_tensors.append(client.model.parameters_to_tensor())
        avg_tensor = sum(p_tensors) / n
        self.center.model.tensor_to_parameters(avg_tensor)
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
                acc = self.center.test_accuracy()
                print('Summary, Accuracy: %.5f, Time: %.0fs' % (
                    acc,
                    time_end - time_begin,
                ))
                self.acc_meter.append(acc)
            print('%sRound %d end%s' % ('=' * 10, round, '=' * 10))
        print('Done, max acc: %.5f' % (self.acc_meter.max()))
