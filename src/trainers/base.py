import torch
import random
import importlib
import numpy as np
from src.trainers.utils import nlp_collate_fn

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
    def __init__(self, id, params, dataset):
        self.batch_size = params['Trainer']['batch_size']
        self.trainset = dataset['train']
        self.testset = dataset['test']
        self.id = id
        collate_fn = None
        dataset_type = 'Image'
        if 'type' in params['Dataset'] and params['Dataset']['type'] == 'NLP':
            dataset_type = 'NLP'
        if dataset_type == 'NLP':
            collate_fn = nlp_collate_fn
        if self.trainset != None:
            self.trainloader = torch.utils.data.DataLoader(
                self.trainset, 
                batch_size=self.batch_size, 
                drop_last=True, 
                shuffle=True,
                collate_fn=collate_fn,
            )
        if self.testset != None:
            self.testloader = torch.utils.data.DataLoader(
                self.testset, 
                batch_size=self.batch_size, 
                drop_last=False, 
                shuffle=True,
                collate_fn=collate_fn,
            )
        self.E = params['Trainer']['E']
        self.device = torch.device(params['Trainer']['device'])
        models = importlib.import_module('src.models')
        self.model = eval('models.%s' % params['Model']['name'])(params)
        if dataset_type == 'NLP':
            self.model.embedding.weight.data.copy_(dataset['vocab'].vectors)
        self.model = self.model.to(self.device)
    
    def local_train(self):
        raise NotImplementedError()
    
    def clone_model(self, target):
        p_tensor = target.model.parameters_to_tensor()
        self.model.tensor_to_parameters(p_tensor)
        return

    def test_accuracy(self):
        if self.testset == None: return -1
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
    
    def get_features_and_labels(self, train=True, batch=-1):
        dataloader = None
        if train: dataloader = self.trainloader
        else: dataloader = self.testloader
        features_batch = []
        labels_batch = []
        with torch.no_grad():
            for i, data in enumerate(dataloader):
                if i == batch: break
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                _, f_s = self.model(inputs, features=True)
                features_batch.append(f_s)
                labels_batch.append(labels)
        features = torch.cat(features_batch)
        labels = torch.cat(labels_batch)
        return features, labels
    
    def save_features_and_labels(self, fn, train=True, batch=-1):
        features, labels = self.get_features_and_labels(train, batch)
        features = features.cpu().numpy()
        labels = labels.cpu().numpy()
        np.save('%s_features.npy' % fn, features)
        np.save('%s_labels.npy' % fn, labels)
        return

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
        self.center = self.Client(0, params, self.testset)
        self.clients = []
        for i in range(self.n_clients):
            self.clients.append(
                self.Client(
                    i + 1, 
                    params,
                    self.dataset_split[i],
                )
            )
        self.learning_rate = self.params['Trainer']['optimizer']['params']['lr']

    def aggregate_model(self):
        raise NotImplementedError()

    def train(self):
        raise NotImplementedError()

    def sample_client(self):
        return random.sample(
            self.clients, 
            self.n_clients_per_round,
        )
