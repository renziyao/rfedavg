import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import time
from src.trainers.base import *
from src.trainers.utils import LinearMMD


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
            'mmd_loss': AvgMeter(),
            'loss': AvgMeter(),
        }
    
    def local_train(self):
        batch_count = 0
        for epoch in range(self.E):
            for i, data in enumerate(self.trainloader):
                self.optimizer.zero_grad()
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs, f_s = self.model(inputs, features=True)
                classifier_loss = self.classifier_criterion(
                    outputs,
                    labels,
                )
                mmd_loss = self.mmd_criterion(f_s, self.f_t)
                loss = classifier_loss + mmd_loss * self.params['Trainer']['lambda']
                loss.backward()
                self.optimizer.step()
                self.meters['classifier_loss'].append(classifier_loss.item())
                self.meters['mmd_loss'].append(mmd_loss.item())
                self.meters['loss'].append(loss.item())
                batch_count += 1
        print('Client %d, classifier_loss: %.5f, mmd_loss: %.5f, loss: %.5f, acc: %.5f' % (
            self.id, 
            self.meters['classifier_loss'].avg(-batch_count),
            self.meters['mmd_loss'].avg(-batch_count),
            self.meters['loss'].avg(-batch_count),
            self.test_accuracy(),
        ), flush=True)

    def get_features(self):
        inputs, _ = next(iter(self.trainloader))
        inputs = inputs.to(self.device)
        _, f_s = self.model(inputs, features=True)
        return f_s


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
            
            f_t = []
            for client in clients:
                # send params
                client.clone_model(self.center)
                f_t.append(client.get_features())
            
            # calc avg features
            f_t_sum = sum(f_t)
            f_t_len = len(f_t)

            # for each client in choose_clients
            for i, client in enumerate(clients):
                client.f_t = (f_t_sum / f_t_len).detach()
                for p in client.optimizer.param_groups:
                    p['lr'] = self.learning_rate
            
            for client in clients:
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
