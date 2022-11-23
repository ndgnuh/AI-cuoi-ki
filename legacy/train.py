import torch
import numpy as np
from accuracy_index import iou
from config import parse_args


def main():
    config = parse_args()
    train_data = config.train_data
    test_data = config.test_data
    model = config.model
    optimizer = config.optimizer
    loss_function = config.loss_function

    # Start training
    lr = optimizer.param_groups[0]["lr"]
    train_size = len(train_data.dataset)
    train_log_throttle = len(train_data) // 5
    test_num_batches = len(test_data)
    accuracies = []
    for t in range(config.start_epoch - 1, config.end_epoch):
        print(f"Epoch {t + 1}")

        # Decay lr every 30 epoch
        optimizer.param_groups[0]["lr"] = lr * \
            (config.decay_rate ** (t // config.decay_every))
        print("Learning rate: ", optimizer.param_groups[0]["lr"])

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
            if batch % train_log_throttle == 0:
                train_loss, train_current = loss.item(), batch * len(X)
                print(f"\tEpoch: {t}, Batch: {batch}, Loss: {train_loss:>7f}  [{train_current:>5d}/{train_size:>5d}]")

        # TEST
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in test_data:
                X = X.to(config.device)
                y = y.to(config.device)
                pred = model(X)
                test_loss += loss_function(pred, y).item()
                correct += iou(pred, y)
        test_loss /= test_num_batches
        correct /= test_num_batches
        accuracies.append(correct)
        print("Test:")
        print(f"\tAccuracy: {(100*correct):>6.2f}%\tAvg loss: {test_loss:>8f}")

        # Save model
        if config.model_path is not None:
            torch.save(model, config.model_path)
            print("Model saved")
            print("")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt by User")
