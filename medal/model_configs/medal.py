import abc
import torch
from contextlib import contextmanager

from .. import checkpointing
from .baseline_inception import BaselineInceptionV3BinaryClassifier
from .baseline_squeezenet import BaselineSqueezeNetBinaryClassifier
from .baseline_resnet18 import BaselineResnet18BinaryClassifier
from . import feedforward


def pick_initial_data_points_to_label(config):
    # TODO: how many?
    return 0


def pick_data_points_to_label(config):
    embedding_labeled, embedding_unlabeled, unlabeled_idxs = \
        get_labeled_and_topk_unlabeled_embeddings(config)

    # centroid of labeled data in euclidean space is the average of all points.
    # but for computational efficiency, maintain two sets and take weighted
    # average.
    N = embedding_labeled.shape[0]  # num previously labeled items
    M = 0  # num newly labeled items
    old_items_centroid = embedding_labeled.mean(0)  # fixed
    new_items_sum = torch.zeros_like(old_items_centroid, device=config.device)
    selected_items_mask = torch.ByteTensor(
        embedding_unlabeled.shape[0]).to(config.device) * 0 + 1
    selected_unlabeled_idxs = []
    # pick unlabeled points, one at a time.  update centroid each time.
    for n in range(config.num_points_to_label_per_al_iter):
        centroid = N/(N+M)*old_items_centroid + 1/(N+M)*new_items_sum

        unlabeled_items = embedding_unlabeled[selected_items_mask]
        dists = torch.norm(unlabeled_items - centroid, p=2, dim=1)

        chosen_point = dists.argmax()
        new_items_sum += unlabeled_items[chosen_point]
        selected_items_mask[chosen_point] = 0
        selected_unlabeled_idxs.append(unlabeled_idxs[chosen_point])

    assert selected_items_mask.shape[0] \
        == config.num_max_entropy_samples
    assert (~selected_items_mask).sum() \
        == config.num_points_to_label_per_al_iter

    # TODO:figure out how to go from a mask of top entropy points
    # to a mask we can join with config.is_labeled.
    #  train_indices[~config.is_labeled].shape[0]  # sanity check
    return selected_unlabeled_idxs


def get_labeled_and_topk_unlabeled_embeddings(config):
    # get model prediction on unlabeled points
    unlabeled_data_loader = feedforward.create_data_loader(
        config, idxs=config.train_indices[~config.is_labeled], shuffle=False)
    labeled_data_loader = feedforward.create_data_loader(
        config, idxs=config.train_indices[config.is_labeled], shuffle=False)

    # get labeled data embeddings
    embedding_labeled, _ = get_feature_embedding(
        config, labeled_data_loader, topk=None)
    # get unlabeled data embeddings on the N highest predictive entropy samples
    embedding_unlabeled, unlabeled_idxs = get_feature_embedding(
        config, unlabeled_data_loader, topk=config.num_max_entropy_samples)
    return embedding_labeled, embedding_unlabeled, unlabeled_idxs


def get_feature_embedding(config, data_loader, topk):
    """Iterate through all items in the data loader and maintain a list
    of top k highest entropy items and their embeddings

    topk - the max number of samples to keep.  If None, don't bother with
    entropy, and just return embeddings for items in the data loader.

    Return the embeddings (topk_points x feature_dimension) and the indexes of
    each embedding in the original data loader.

    - Only 1 forward pass to get entropy and feature embedding
    - Done in a streaming fashion to be ram conscious
    """
    config.model.eval()
    _batched_embeddings = []
    with torch.no_grad(), register_embedding_hook(
            config.get_feature_embedding_layer(), _batched_embeddings):
        entropy = torch.tensor([]).to(config.device)
        embeddings = []
        loader_idxs = []
        N = 0
        for X, y in data_loader:
            # get entropy and embeddings for this batch
            X, y = X.to(config.device), y.to(config.device)
            yhat = config.model(X)
            embeddings.extend(_batched_embeddings.pop())
            assert len(_batched_embeddings) == 0  # sanity check forward hook
            loader_idxs.extend(range(X.shape[0]))
            # select only top k values
            if topk is not None:
                _entropy = -yhat*torch.log2(yhat) - (1-yhat)*torch.log2(1-yhat)
                entropy = torch.cat([entropy, _entropy])
                if len(entropy) > topk:
                    entropy, idxs = torch.topk(entropy, topk, dim=0)
                    embeddings = [embeddings[i] for i in idxs]
                    loader_idxs = [loader_idxs[i] for i in idxs]
            N += X.shape[0]

        embeddings = torch.stack(embeddings)
        embeddings = embeddings.reshape(embeddings.shape[0], -1)
        return embeddings.detach(), loader_idxs


@contextmanager
def register_embedding_hook(layer, output_arr):
    """
    Temporarily add a hook to a pytorch layer to capture output of that layer
    on forward pass

        >>> myemptylist = []
        >>> layer = next(model.children())  # pick any layer from the model
        >>> with register_embedding_hook(layer, myemptylist):
        >>>     model(X)
        >>> # now myemptylist is populated with output of given layer
    """
    handle = layer.register_forward_hook(
        lambda thelayer, inpt, output: output_arr.append(output)
    )
    yield
    handle.remove()


def train(config):
    """Train a feedforward network using MedAL method"""

    for al_iter in range(config.cur_al_iter + 1, config.al_iters + 1):
        # update state for new al iteration
        config.cur_epoch = 0
        config.cur_al_iter = al_iter

        # train model the regular way
        config.train_loader = feedforward.create_data_loader(
            config, idxs=config.train_indices[config.is_labeled])
        feedforward.train(config)  # train for many epochs

        # pick unlabeled points to label
        mask = pick_data_points_to_label(config)
        config.train_indices[~config.is_labeled][mask] = 1
        print("HELLO WORLD", config.labeled.sum())

        #  reset_weights  # TODO


class MedalConfigABC(feedforward.FeedForwardModelConfig):
    """Base class for all MedAL models"""
    run_id = str
    al_iters = int

    num_max_entropy_samples = int
    num_points_to_label_per_al_iter = int

    @abc.abstractmethod
    def get_feature_embedding_layer(self):
        raise NotImplementedError

    checkpoint_fname = \
        "{config.run_id}/al_{config.cur_al_iter}_epoch_{config.cur_epoch}.pth"
    cur_al_iter = 0  # it's actually 1 indexed

    def train(self):
        return train(self)

    def load_checkpoint(self, check_loaded_all_available_data=True):
        extra_state = super().load_checkpoint()
        # ensure loaded right checkpoint
        # same processing that feedforward does for epoch.
        if self.cur_al_iter != 0:
            checkpointing.ensure_consistent(
                extra_state, key='al_iter', value=self.cur_al_iter)
        elif extra_state is None:  # no checkpoint found
            return

        self.cur_al_iter = extra_state.pop('al_iter')
        if check_loaded_all_available_data:
            assert len(extra_state) == 0, extra_state
        return extra_state

    def __init__(self, config_override_dict):
        super().__init__(config_override_dict)

        # override the default feedforward config
        self.log_msg_minibatch = \
            "--> al_iter {config.cur_al_iter} " + self.log_msg_minibatch[4:]
        self.log_msg_epoch = \
            "al_iter {config.cur_al_iter} " + self.log_msg_epoch

        # split train set into unlabeled and labeled points
        self.train_indices = torch.tensor(
            self.train_loader.sampler.indices.copy(),
            dtype=torch.long, device=self.device)
        del self.train_loader  # will recreate appropriately during train.
        self.is_labeled = torch.ByteTensor(
            self.train_indices.shape).to(self.device) * 0

        mask = pick_initial_data_points_to_label(self)
        self.is_labeled[mask] = 1


class MedalInceptionV3BinaryClassifier(MedalConfigABC,
                                       BaselineInceptionV3BinaryClassifier):
    run_id = 'medal_inceptionv3'
    al_iters = 34

    num_max_entropy_samples = 20
    num_points_to_label_per_al_iter = 10

    def get_feature_embedding_layer(self):
        return list(self.model.children())[0][7]


class MedalSqueezeNetBinaryClassifier(MedalConfigABC,
                                      BaselineSqueezeNetBinaryClassifier):
    run_id = 'medal_squeezenet'
    al_iters = 34

    num_max_entropy_samples = 20
    num_points_to_label_per_al_iter = 5

    def get_feature_embedding_layer(self):
        return list(self.model.children())[0][0][6]


class MedalResnet18BinaryClassifier(MedalConfigABC,
                                    BaselineResnet18BinaryClassifier):
    run_id = 'medal_resnet18'
    al_iters = 34

    num_max_entropy_samples = 20
    num_points_to_label_per_al_iter = 5

    def get_feature_embedding_layer(self):
        return list(self.model.children())[0][0][6]
