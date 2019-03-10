"""
Functions to save and load model checkpoints to/from disk
"""
import glob
import os
from os.path import dirname
import torch


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
