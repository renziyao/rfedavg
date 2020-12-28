import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import dalib.adaptation.dan
import dalib.modules.kernels
import random
from src.model import *
from src.dataloader import *
from src.utils import *

def main(params):
    device = params['device']

    # generate center model
    center_model = LeNet()
    center_model.to(device)
    center_model.train()

    # generate client model and datasets
    n_clients = params['n_clients']
    client_ids = list(range(n_clients))
    client_models = [LeNet() for i in client_ids]
    for model in client_models:
        model.to(device)
        model.train()
    print('generated model ... Done')

    client_datasets, testset = get_mnist_train_niid('./data/', n_clients)
    print('generated data ... Done')

    # generate criterion
    classifier_criterion = nn.CrossEntropyLoss()
    mmd_criterion = LinearMMD()
    print('generated criterion ... Done')

    # define local train func
    def local_train(model, dataset_source, f_t):
        meters = {'classifier_loss': [], 'mmd_loss': []}
        optimizer = optim.SGD(
            [
                {'params': model.parameters()},
            ], 
            lr=params['optimizer']['lr'],
            momentum=params['optimizer']['momentum'],
            weight_decay=params['optimizer']['weight_decay'],
        )
        trainloader = torch.utils.data.DataLoader(
            dataset_source, 
            batch_size=params['batch_size'], 
            drop_last=True, 
            shuffle=True,
        )
        
        for epoch in range(params['E']):
            for i, data in enumerate(trainloader):
                optimizer.zero_grad()
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs, f_s = model(inputs)
                classifier_loss = classifier_criterion(
                    outputs,
                    labels,
                )
                mmd_loss = mmd_criterion(f_s, f_t)
                loss = classifier_loss + mmd_loss * params['lambda']
                loss.backward()
                optimizer.step()
                meters['classifier_loss'].append(classifier_loss.item())
                meters['mmd_loss'].append(mmd_loss.item())
            classifier_loss = sum(meters['classifier_loss']) / len(meters['classifier_loss'])
            mmd_loss = sum(meters['mmd_loss']) / len(meters['mmd_loss'])
            if params['DEBUG']:
                print('client, classifier_loss: %.5f, mmd_loss: %.5f' % (
                    classifier_loss, 
                    mmd_loss,
                    ), flush=True)
        return

    def get_local_features(model, dataset):
        trainloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=params['batch_size'], 
            drop_last=True, 
            shuffle=True,
        )
        inputs, _ = next(iter(trainloader))
        inputs = inputs.to(device)
        _, f_s = model(inputs)
        return f_s
    
    for comm_round in range(params['T']):
        print('communication round %d' % comm_round)
        # test if needed
        if comm_round % params['TestInterval'] == 0 or \
            comm_round == params['T'] - 1:
            center_model.train(False)
            acc_target = test_acc(center_model, testset, device)
            print("target acc: %.5f" % (acc_target))
            center_model.train(True)
        
        f_t = []
        for client_id in client_ids:
            f_t.append(
                get_local_features(
                    client_models[client_id], 
                    client_datasets[client_id]
                )
            )
        f_t = (sum(f_t) / len(f_t)).detach()

        # random clients
        choose_client = random.sample(client_ids, round(params['C'] * n_clients))
        
        # for each client in choose_clients
        for client_id in choose_client:
            # get client model
            client_model = client_models[client_id]

            # send params
            send_params(center_model, client_model)
            
            # local train
            local_train(client_model, client_datasets[client_id], f_t)

        # aggregate params
        aggregate_params(center_model, client_models)
