import torch
import numpy as np
from torch import nn
from torch import optim
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from pprint import pprint

# our modules
from models import MLP
import models

def load_data(**kwargs):
	dataset = CIFAR100("dataset/cifar100", download=True, **kwargs)
	return DataLoader(dataset, batch_size=100, shuffle=True)

def train_loop(data, model, floss, opt):
	size = len(data.dataset)
	for batch, (X, y) in enumerate(data):
		yhat = model(X)
		loss = floss(yhat, y)
		opt.zero_grad()
		loss.backward()
		opt.step()
		if batch % 100 == 0:
			loss, current = loss.item(), batch * len(X)
			print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
	return loss


def test_loop(dataloader, model, loss_fn):
	size = len(dataloader.dataset)
	num_batches = len(dataloader)
	test_loss, correct = 0, 0
	with torch.no_grad():
		for X, y in dataloader:
			pred = model(X)
			test_loss += loss_fn(pred, y).item()
			correct += (pred.argmax(1) == y).type(torch.float).sum().item()
	test_loss /= num_batches
	correct /= size
	print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
	return test_loss, correct

def adjust_learning_rate(lr, optimizer, epoch):
	"""
	Sets the learning rate to the initial LR decayed by 10 every 30 epochs
	"""
	lr = lr * (0.1 ** (epoch // 30))
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr
	return lr

def main(config):
	device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
	from math import pi
	train_data = load_data(train=True, transform=ToTensor())
	test_data = load_data(train=False, transform=ToTensor())
	checkpoint = 'checkpoint/model.pth'

	model = None
	try:
		model = torch.load(checkpoint)
		print("Model loaded")
	except:
		model = MLP(3072, 100).to(device)
	print(model)

	epochs = 1000000
	current_epoch = 0
	loss_fn = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(), lr=config.lr)
	for t in range(config.start_epoch, epochs):
		print(f"Epoch ${t + 1}")
		# if previous_loss is not None and np.abs((previous_loss - loss).item()) < 1e-5:
		# 	lr = lr / 10
		# 	print(f"Reduced learning rate to {lr}")
		# previous_loss = loss
		loss = train_loop(train_data, model, loss_fn, optimizer)
		tloss, correct = test_loop(test_data, model, loss_fn)
		torch.save(model, checkpoint)
		lr_ = adjust_learning_rate(lr, optimizer, t)
		print("Learning rate: ", lr_)
		if correct > 0.9 or loss < 0.05:
			break
		print("Model saved")


def list_model():
	print("[LIST OF MODELS]")
	for k in modelmap:
		print("\t", k, "--", modelmap[k])
	print()

def main_wip():
	from config import parser, parse_args
	config = parse_args()

	if config.list_model:
		list_model()
		return 0

	config.model = find_model(config.model)
	if config.run_old_main:
		print("Learning rate: ", config.lr)
		main(lr = float(config.lr), start_epoch=int(config.start_epoch))
		return 0

	model = find_model(config.model)
	print("Selected model: ", model)
	print("Learning rate: ", config.lr, type(config.lr))


if __name__ == '__main__':
	print("This file is deprecated, run train.py instead")
