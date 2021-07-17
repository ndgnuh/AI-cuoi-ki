import torch
import numpy as np
from torch import nn
from torch import optim
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from pprint import pprint
from config import parser


def main():
	config = parser.parse_args()

	train_data = CIFAR100("dataset/cifar100",
			download=True,
			train=True,
			transform=ToTensor())
	train_data = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
	test_data = CIFAR100("dataset/cifar100",
			download=True,
			train=False, # Tội copy paste
			transform=ToTensor())
	test_data = DataLoader(test_data, batch_size=config.batch_size, shuffle=True)

	# Start training
	model = config.model
	lr = config.lr
	loss_function = nn.CrossEntropyLoss()
	train_size = len(train_data.dataset)
	test_size = len(test_data.dataset)
	test_num_batches = len(test_data)
	print(model)
	for t in range(config.start_epoch - 1, config.end_epoch):
		print(f"Epoch {t + 1}")

		# Decay lr every 30 epoch
		lr = config.lr * (config.decay_rate ** (t // config.decay_every))
		optimizer = optim.Adam(config.model.parameters(), lr=lr)
		print("Learning rate: ", lr)

		# TRAIN
		train_loss = None
		for batch, (X, y) in enumerate(train_data):
			yhat = model(X)
			loss = loss_function(yhat, y)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			if batch % 100 == 0:
				train_loss, train_current = loss.item(), batch * len(X)
				print(f"\tLoss: {train_loss:>7f}  [{train_current:>5d}/{train_size:>5d}]")

		# TEST
		test_loss, correct = 0, 0
		with torch.no_grad():
			for X, y in test_data:
				pred = model(X)
				test_loss += loss_function(pred, y).item()
				correct += (pred.argmax(1) == y).type(torch.float).sum().item()
		test_loss /= test_num_batches
		correct /= test_size
		print(f"Test:")
		print(f"\tAccuracy: {(100*correct):>0.1f}%")
		print(f"\tAvg loss: {test_loss:>8f}")


		# Save model
		if config.model_path is not None:
			torch.save(model, config.model_path)
		print("Model saved\n")
		if correct > 0.9 or loss < 0.05:
			break

if __name__ == '__main__':
	try:
		main()
	except KeyboardInterrupt:
		print("KeyboardInterrupt by User")
