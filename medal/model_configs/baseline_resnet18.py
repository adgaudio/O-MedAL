from os.path import join
import torch
import torch.optim
import torchvision.transforms as tvt

from . import feedforward
from .. import datasets
from .. import models


class BaselineResnet18BinaryClassifier(feedforward.FeedForwardModelConfig):
    epochs = 100
    batch_size = 48
    learning_rate = 0.001
    train_frac = .8
    weight_decay = 0.01
    trainable_resnet_layers = True
    trainable_top_layers = True
    load_pretrained_resnet18_weights = True

    def get_model(self):
        model = models.Resnet18BinaryClassifier(self)
        model.set_layers_trainable(
            resnet18_layers=self.trainable_resnet_layers,
            top_layers=self.trainable_top_layers)
        return model

    def get_lossfn(self):
        return torch.nn.modules.loss.BCEWithLogitsLoss()

    def get_optimizer(self):
        return torch.optim.SGD(
            self.model.parameters(),
            momentum=0.9, lr=self.learning_rate,
            weight_decay=self.weight_decay, nesterov=True)

        #  return torch.optim.Adam(
            #  self.model.parameters(), lr=self.learning_rate, eps=0.1,
            #  weight_decay=self.weight_decay, betas=(.9, .999))

    def get_dataset(self):
        return datasets.Messidor(
            join(self.base_dir, "messidor/*.csv"),
            join(self.base_dir, "messidor/**/*.tif"),
            img_transform=tvt.Compose([
                tvt.RandomRotation(degrees=15),
                tvt.RandomResizedCrop(
                    512, scale=(0.9, 1.0), ratio=(1, 1)),
                tvt.RandomHorizontalFlip(),
                #  tvt.RandomVerticalFlip(),
                tvt.ToTensor(),
            ]),
            getitem_transform=lambda x: (
                x['image'],
                torch.tensor([float(x['Retinopathy grade'] != 0)]))
        )

    def get_data_loaders(self):
        train_idxs, val_idxs = self.dataset.train_test_split(
            train_frac=self.train_frac)
        return (
            feedforward.create_data_loader(self, train_idxs),
            feedforward.create_data_loader(self, val_idxs))
