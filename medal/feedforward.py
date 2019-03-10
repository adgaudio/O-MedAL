"""
Functions to train and test feedforward networks using backprop
"""
import torch

from .checkpointing import save_checkpoint


def train_one_epoch(config, epoch):
    config.model.train()
    train_loss, train_correct, N = 0, 0, 0
    for batch_idx, (X, y) in enumerate(config.train_loader):
        if X.shape[0] != config.batch_size:
            #  print("Skipping end of batch", X.shape)
            continue
        X, y = X.to(config.device), y.to(config.device)
        config.optimizer.zero_grad()
        yhat = config.model(X)
        loss = config.lossfn(yhat, y.float())
        loss.backward()
        config.optimizer.step()

        with torch.no_grad():
            batch_size = X.shape[0]
            _loss = loss.item() * batch_size
            train_loss += _loss
            _correct = y.int().eq((yhat.view_as(y) > .5).int()).sum().item()
            train_correct += _correct
            N += batch_size

            # print output if batch_idx % config.log_interval == 0
            if batch_idx % 10 == 0:
                print(
                    '-->', 'epoch:', epoch, 'batch_idx', batch_idx,
                    'train_loss:', train_loss/N,
                    'train_acc', train_correct / N)
    return train_loss/N, train_correct/N


def train(config, epoch):
    for epoch in range(epoch, config.epochs + 1):
        train_loss, train_acc = train_one_epoch(config, epoch)
        save_checkpoint(config, epoch)
        val_loss, val_acc = test(config)
        print(
            "epoch", epoch, "train_loss", train_loss, "val_loss", val_loss,
            "train_acc", train_acc, "val_acc", val_acc)


def test(config):
    """Return avg loss and accuracy on the validation data"""
    config.model.eval()
    totloss = 0
    correct = 0
    N = 0
    with torch.no_grad():
        for X, y in config.val_loader:
            batch_size = X.shape[0]
            X, y = X.to(config.device), y.to(config.device)
            yhat = config.model(X)
            totloss += (config.lossfn(yhat, y.float()) * batch_size).item()
            correct += y.int().eq((yhat.view_as(y) > .5).int()).sum().item()
            N += batch_size
    return totloss/N, correct/N
