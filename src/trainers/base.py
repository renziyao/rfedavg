import numpy as np
import torch
import random
import importlib

class AvgMeter():
    def __init__(self):
        self.data = []
    
    def append(self, x, times=1):
        for _ in range(times):
            self.data.append(x)
        return

    def avg(self, p=0):
        return sum(self.data[p:]) / len(self.data[p:])
    
    def min(self, p=0):
        return min(self.data[p:])
    
    def max(self, p=0):
        return max(self.data[p:])
    
    def last(self):
        return self.data[-1]

class BaseClient():
    def __init__(self, id, params, trainset, testset):
        self.BATCH_SIZE = params['Trainer']['batch_size']
        self.trainset = trainset
        self.testset = testset
        self.id = id
        if trainset != None:
            self.trainloader = torch.utils.data.DataLoader(
                trainset, 
                batch_size=self.BATCH_SIZE, 
                drop_last=True, 
                shuffle=True,
            )
        if testset != None:
            self.testloader = torch.utils.data.DataLoader(
                testset, 
                batch_size=self.BATCH_SIZE, 
                drop_last=False, 
                shuffle=True,
            )
        self.E = params['Trainer']['E']
        self.device = torch.device(params['Trainer']['device'])
        models = importlib.import_module('src.models')
        self.model = eval('models.%s' % params['General']['model'])()
        self.model = self.model.to(self.device)
    
    def train():
        raise NotImplementedError()

    def clone_model(self, target):
        target_params = target.model.state_dict()
        for (name, params) in self.model.named_parameters():
            params.data = target_params[name].clone().detach()
        return

    def aggregate_model(self, client_models):
        n = len(client_models)
        with torch.no_grad():
            for (name, params) in self.model.named_parameters():
                params.data = torch.zeros_like(params.data)
                for i, client_model in enumerate(client_models):
                    params.data += (1 / n) * \
                        client_model.model.state_dict()[name]
        return

    def test_accuracy(self):
        correct = 0
        total = 0
        with torch.no_grad():
            for data in self.testloader:
                images, labels = data
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return correct / total

class BaseServer():
    def __init__(self, params):
        self.T = params['Trainer']['T']
        self.TEST_INTERVAL = params['Trainer']['test_interval']
        self.n_clients = params['Trainer']['n_clients']
        self.n_clients_per_round = round(params['Trainer']['C'] * self.n_clients)
        dataset_name, func_name = params['General']['dataset'].split('.')
        dataset_module = importlib.import_module(
            'src.data.%s' % dataset_name
        )
        dataset_func = eval('dataset_module.%s' % func_name)
        dataset_split, testset = dataset_func(params)
        self.dataset_split = dataset_split
        self.testset = testset
        # load clients

    def train(self):
        selected_clients = self.select_client()
        for client in selected_clients:
            self.send_params(self.models[client])
            self.local_train(self.model[client], self.dataloader[client])
        self.aggregate_params(self.center_model, self.models)
        return

    def sample_client(self):
        return random.sample(
            self.clients, 
            self.n_clients_per_round,
        )
