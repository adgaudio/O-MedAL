"""
Functions to save and load model checkpoints to/from disk
"""
import glob
import os
from os.path import dirname, join
import torch


def _get_checkpoint_fp(config):
    return join(config.checkpoint_dir, config.checkpoint_fname).format(
        config=config)


def save_checkpoint(config, extra_state=None):
    """Save a model checkpoint to disk.
    By default, save only the model and optimizer.

    config - an object with these attributes:
        config.checkpoint_fname
        config.checkpoint_dir
    extra_state - a dict with additional data to store in the checkpoint file.
    """
    save_fp = _get_checkpoint_fp(config)

    os.makedirs(dirname(save_fp), exist_ok=True)
    state = {
        'model_state_dict': config.model.state_dict(),
        'optimizer_state_dict': config.optimizer.state_dict(),
    }
    state.update(extra_state or {})
    print("Save checkpoint", save_fp)
    torch.save(state, save_fp)


def load_checkpoint(config):
    """Load a model from disk.
    config - an object with these attributes:
        config.checkpoint_fname  (ie "epoch_{config.epoch}_runid_{run_id}.pth")
        config.checkpoint_dir

    This function will only restore the model and optimizer.
    If other data is present, it will be returned

    If multiple filepaths match, fail.
    """
    read_fp = _get_checkpoint_fp(config)
    fps = glob.glob(read_fp)
    if fps:  # yay - there is a checkpoint to restore
        if len(fps) != 1:
            raise Exception(
                "Too many checkpoint files.  Don't know how to restore."
                " Files are: \n%s" % '\n'.join(fps))
        assert len(fps) == 1
        fp = max(fps)
        print("Restoring from checkpoint:", fp)
        checkpoint = torch.load(fp)
        config.model.load_state_dict(checkpoint.pop('model_state_dict'))
        config.optimizer.load_state_dict(
            checkpoint.pop('optimizer_state_dict'))
        return checkpoint
    else:
        print("Did not restore a checkpoint.  No file matches: %s" % read_fp)
        return


def ensure_consistent(extra_state, key, value):
    """Check the state returned by load_checkpoint has given key and value
    Raise an error if it is not consistent
    """
    if not isinstance(extra_state, dict):
        raise Exception(
            "Could not find requested checkpoint at %s: %s" % (key, value))
    assert extra_state[key], "bug: loaded incorrect checkpoint data"
    if extra_state[key] != value:
        raise Exception((
            "bug: Data inconsistency! %s stored in checkpoint"
            " does not match file loaded.") % key)
