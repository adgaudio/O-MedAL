import configargparse as ap
import glob
from os.path import dirname
import os
import torch

from . import model_configs as MC

#  def MNIST_get_data_loaders(config):
#      """For debugging MedAL code, used the MNIST dataset"""
#      trans = tvt.Compose([
#          tvt.Grayscale(3), tvt.ToTensor(), tvt.Normalize((0.5,), (1.0,)),
#      ])
#      ttrans = tvt.Compose([
#          lambda x: x.reshape(1, 1) > 5
#      ])
#      train_set = tv.datasets.MNIST(
#          root=join(DATA_DIR, "MNIST"),
#          train=True, transform=trans, target_transform=ttrans, download=True)
#      test_set = tv.datasets.MNIST(
#          root=join(DATA_DIR, "MNIST"),
#          train=False, transform=trans, target_transform=ttrans,
#          download=True)

#      for x in train_set, test_set:
#          yield torch.utils.data.DataLoader(
#                          dataset=train_set,
#                          batch_size=config.batch_size,
#                          shuffle=True)


CHECKPOINT_FP_TEMPLATE = "{checkpoint_dir}/{run_id}/epoch_{epoch}.pth"


def save_checkpoint(config, epoch):
    save_fp = CHECKPOINT_FP_TEMPLATE.format(
        checkpoint_dir=config.checkpoint_dir,
        run_id=config.run_id, epoch=epoch)

    os.makedirs(dirname(save_fp), exist_ok=True)
    print("Save checkpoint", save_fp)
    torch.save({
        'epoch': epoch,
        'model_state_dict': config.model.state_dict(),
        'optimizer_state_dict': config.optimizer.state_dict(),
    }, save_fp)


def load_checkpoint(config):
    read_fp = CHECKPOINT_FP_TEMPLATE.format(
        checkpoint_dir=config.checkpoint_dir,
        run_id=config.run_id, epoch='*')
    fps = glob.glob(read_fp)
    if fps:  # yay - there is a checkpoint to restore
        fp = max(fps)
        print("Restoring from checkpoint:", fp)
        checkpoint = torch.load(fp)
        config.model.load_state_dict(checkpoint['model_state_dict'])
        config.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
    else:
        epoch = 1
    return epoch


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


def main(ns: ap.Namespace):
    config_overrides = ns.__dict__
    config = config_overrides.pop('modelconfig_class')(config_overrides)
    # define the dataset
    print('\n'.join(str((k, v)) for k, v in config.__dict__.items()
                    if not k.startswith('__')))

    if config.device == 'cuda' and torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        config.model = torch.nn.DataParallel(config.model)
    config.model.to(config.device)

    epoch = load_checkpoint(config)
    train(config, epoch)


def _add_subparser_find_configurable_attributes(kls):
    # get a list of configurable attributes from given class and parent classes
    keys = set()
    klass_seen = set()
    klass_lst = [kls]
    while klass_lst:
        klass = klass_lst.pop()
        for k in klass.__bases__:
            if k in klass_seen:
                continue
            klass_seen.add(k)
            klass_lst.append(k)
        keys.update({x for x in klass.__dict__ if not x.startswith('_')})
    return keys


def _add_subparser_arg(subparser, k, v, obj):
    """Add an argparse option via subparser.add_argument(...)
    for the given attribute k, with default value v, and where hasattr(obj, k)
    """
    g = subparser
    accepted_simple_types = (int, float, str)
    ku = k.replace('_', '-')
    if isinstance(v, bool):
        grp = g.add_mutually_exclusive_group()
        grp.add_argument('--%s' % ku, action='store_const', const=True)
        grp.add_argument(
            '--no-%s' % ku, action='store_const', const=False, dest=k)
    elif isinstance(v, accepted_simple_types):
        g.add_argument('--%s' % ku, type=type(v),
                       default=getattr(obj, k))
    elif isinstance(v, (list, tuple)):
        if all(isinstance(x, accepted_simple_types) for x in v):
            g.add_argument(
                '--%s' % ku,
                nargs=len(v), type=type(v[0]))
        else:
            g.add_argument(
                '--%s' % ku, nargs=len(v), type=v[0])
    elif any(v is x for x in accepted_simple_types):
        g.add_argument('--%s' % ku, type=v)


def add_subparser(subparsers, name: str, modelconfig_class: type):
    """
    Automatically add argument parser options for attributes in the class and
    all parent classes
    """
    g = subparsers.add_parser(name)
    g.add_argument(
        '--modelconfig_class', help=ap.SUPPRESS, default=modelconfig_class)

    # add an argument for each configurable key that we can work with
    keys = _add_subparser_find_configurable_attributes(modelconfig_class)
    for k in keys:
        v = getattr(modelconfig_class, k)
        _add_subparser_arg(g, k, v, modelconfig_class)


def build_arg_parser():
    p = ap.ArgumentParser()
    sp = p.add_subparsers(help='model', required=True)
    add_subparser(sp, 'baseline-inception', MC.BaselineInceptionV3)
    return p


if __name__ == "__main__":
    main(build_arg_parser().parse_args())
