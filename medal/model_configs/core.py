from os.path import join
import abc
import torch
import torch.optim

from .. import feedforward


class FeedForwardModelConfig(abc.ABC):
    run_id = str

    batch_size = int
    epochs = int

    base_dir = './data'
    checkpoint_dir = str
    torch_model_dir = str

    model = NotImplemented
    optimizer = NotImplemented
    lossfn = NotImplemented
    dataset = NotImplemented
    train_loader = NotImplemented
    val_loader = NotImplemented

    def train(self, epoch):
        return feedforward.train(self, epoch)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint_interval = 1  # save checkpoint during training every N epochs

    def __init__(self, config_override_dict):
        self.__dict__.update({k: v for k, v in config_override_dict.items()
                              if v is not None})
        self.checkpoint_dir = join(self.base_dir, 'model_checkpoints')
        self.torch_model_dir = join(self.base_dir, 'torch/models')

    def __repr__(self):
        return "config:%s" % self.run_id
