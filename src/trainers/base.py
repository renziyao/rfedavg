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
        self.batch_size = params['Trainer']['batch_size']
        self.trainset = trainset
        self.testset = testset
        self.id = id
        if trainset != None:
            self.trainloader = torch.utils.data.DataLoader(
                trainset, 
                batch_size=self.batch_size, 
                drop_last=True, 
                shuffle=True,
            )
        if testset != None:
            self.testloader = torch.utils.data.DataLoader(
                testset, 
                batch_size=self.batch_size, 
                drop_last=False, 
                shuffle=True,
            )
        self.E = params['Trainer']['E']
        self.device = torch.device(params['Trainer']['device'])
        models = importlib.import_module('src.models')
        self.model = eval('models.%s' % params['Model']['name'])(params)
        self.model = self.model.to(self.device)
    
    def local_train(self):
        raise NotImplementedError()
    
    def clone_model(self, target):
        p_tensor = target.model.parameters_to_tensor()
        self.model.tensor_to_parameters(p_tensor)
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
        self.device = torch.device(params['Trainer']['device'])
        self.Round = params['Trainer']['Round']
        self.test_interval = params['Trainer']['test_interval']
        self.n_clients = params['Trainer']['n_clients']
        self.n_clients_per_round = round(params['Trainer']['C'] * self.n_clients)
        dataset_name = params['Dataset']['name']
        func_name = params['Dataset']['divide']
        dataset_module = importlib.import_module(
            'src.data.%s' % dataset_name
        )
        dataset_func = eval('dataset_module.%s' % func_name)
        dataset_split, testset = dataset_func(params)
        self.dataset_split = dataset_split
        self.testset = testset
        self.params = params
        self.acc_meter = AvgMeter()

    def aggregate_model(self):
        raise NotImplementedError()

    def train(self):
        raise NotImplementedError()

    def sample_client(self):
        return random.sample(
            self.clients, 
            self.n_clients_per_round,
        )
