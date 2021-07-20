import torch
import numpy as np
from torch import nn
from torch import optim
from config import parser


def main():
    config = parser.parse_args()
    train_data = config.train_data
    test_data = config.test_data

    # Start training
    model = config.model
    lr = config.lr
    loss_function = config.loss_function
    train_size = len(train_data.dataset)
    test_size = len(test_data.dataset)
    test_num_batches = len(test_data)
    model = model.to(config.device)
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
            X = X.to(config.device)
            y = y.to(config.device)
            yhat = model(X)
            loss = loss_function(yhat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if batch % 100 == 0:
                train_loss, train_current = loss.item(), batch * len(X)
                print(
                    f"\tLoss: {train_loss:>7f}  [{train_current:>5d}/{train_size:>5d}]")

        # TEST
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in test_data:
                X = X.to(config.device)
                y = y.to(config.device)
                pred = model(X)
                test_loss += loss_function(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= test_num_batches
        correct /= test_size
        print(
            f"Test:\tAccuracy: {(100*correct):>0.1f}%\t\tAvg loss: {test_loss:>8f}")

        # Save model
        if config.model_path is not None:
            torch.save(model, config.model_path)
            print("Model saved\n")
        if correct > 0.9:
            break


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt by User")
