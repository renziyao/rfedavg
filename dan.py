import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import dalib.adaptation.dan
import dalib.modules.kernels
from network import AlexNet
from dataset import generate_dataset
from utils import test_acc, ForeverDataIterator

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

trainset_source, trainset_target, testset_source, testset_target = generate_dataset()

trainloader_source = torch.utils.data.DataLoader(
    trainset_source, batch_size=36, drop_last=True, shuffle=True)
trainloader_target = torch.utils.data.DataLoader(
    trainset_target, batch_size=36, drop_last=True, shuffle=True)
net = AlexNet()
net.to(device)
optimizer = optim.SGD([
    {'params': net.features[8:].parameters(), 'lr': 0.0003},
    {'params': net.fc.parameters()},
    {'params': net.classifier.parameters()},
], lr=0.003, momentum=0.9, weight_decay=0.0005, nesterov=True)
classifier_criterion = nn.CrossEntropyLoss()
mmd_criterion = dalib.adaptation.dan.MultipleKernelMaximumMeanDiscrepancy(
    kernels=[dalib.modules.kernels.GaussianKernel(alpha=2 ** k)
        for k in range(-3, 2)],
    quadratic_program=True,
)
source_iter = ForeverDataIterator(trainloader_source)
target_iter = ForeverDataIterator(trainloader_target)

meters = {'classifier_loss': [], 'mmd_loss': []}
net.train()
for epoch in range(20000):
	optimizer.zero_grad()
	source_inputs, labels = next(source_iter)
	source_inputs = source_inputs.to(device)
	labels = labels.to(device)
	target_inputs, _ = next(target_iter)
	target_inputs = target_inputs.to(device)
	inputs_all = torch.cat((source_inputs, target_inputs), dim=0)
	outputs, features = net(inputs_all)
	classifier_loss = classifier_criterion(
		outputs.narrow(0, 0, source_inputs.shape[0]),
		labels,
	)
	mmd_loss = []
	for item in features:
		mmd_loss.append(mmd_criterion(
			item.narrow(0, 0, source_inputs.shape[0]),
			item.narrow(0, source_inputs.shape[0], source_inputs.shape[0]),
		))
	mmd_loss = sum(mmd_loss)
	tradeoff = 1.0
	loss = classifier_loss + mmd_loss * tradeoff
	loss.backward()
	optimizer.step()
	meters['classifier_loss'].append(classifier_loss.item())
	meters['mmd_loss'].append(mmd_loss.item())
	if epoch % 1 == 0:
		classifier_loss = sum(meters['classifier_loss']) / len(meters['classifier_loss'])
		mmd_loss = sum(meters['mmd_loss']) / len(meters['mmd_loss'])
		print('[%d] classifier_loss: %.3f, mmd_loss: %.3f' %
				(epoch + 1, classifier_loss, mmd_loss), flush=True)
		meters['classifier_loss'] = []
		meters['mmd_loss'] = []
		net.train(False)
		test_acc(net, testset_source, device)
		test_acc(net, testset_target, device)
		net.train(True)
