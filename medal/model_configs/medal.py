import abc
import pickle
import torch
from contextlib import contextmanager

from .. import checkpointing
from .baseline_inception import BaselineInceptionV3BinaryClassifier
from .baseline_squeezenet import BaselineSqueezeNetBinaryClassifier
from .baseline_resnet18 import BaselineResnet18BinaryClassifier
from . import feedforward


def pick_initial_data_points_to_label(config):
    # TODO: better initialization like used in paper
    return torch.randint(
        0, len(config._train_indices),
        (config.num_points_to_label_per_al_iter, ))


def pick_data_points_to_label(config):
    """Return indexes of unlabeled points to sample next.

    The index is over the array formed by the set of all unlabeled points.
    """
    embedding_labeled, embedding_unlabeled, unlabeled_idxs = \
        get_labeled_and_topk_unlabeled_embeddings(config)

    if unlabeled_idxs.shape[0] <= config.num_points_to_label_per_al_iter:
        return unlabeled_idxs

    # centroid of labeled data in euclidean space is the average of all points.
    # but for computational efficiency, maintain two sets and take weighted
    # average.
    N = embedding_labeled.shape[0]  # num previously labeled items
    M = 0  # num newly labeled items
    old_items_centroid = embedding_labeled.mean(0)  # fixed
    new_items_sum = torch.zeros_like(old_items_centroid, device=config.device)
    remaining_unpicked_items = torch.ByteTensor(
        embedding_unlabeled.shape[0]).to(config.device) * 0 + 1
    points_to_label = torch.empty(
        config.num_points_to_label_per_al_iter,
        dtype=torch.long, device=config.device)

    # pick unlabeled points, one at a time.  update centroid each time.
    for n in range(config.num_points_to_label_per_al_iter):
        centroid = N/(N+M)*old_items_centroid + 1/(N+M)*new_items_sum

        unlabeled_items = embedding_unlabeled[remaining_unpicked_items]
        dists = torch.norm(unlabeled_items - centroid, p=2, dim=1)

        chosen_point = dists.argmax()
        new_items_sum += unlabeled_items[chosen_point]
        #  assert unlabeled_idxs.shape[0] == embedding_unlabeled.shape[0]
        points_to_label[n] = \
            unlabeled_idxs[remaining_unpicked_items][chosen_point]

        # --> update the list of remaining unpicked items.
        # This is complicated because r[r][chosen_point] = 0 doesn't work :(
        _tmp = torch.arange(
            remaining_unpicked_items.shape[0], device=config.device)
        _tmp2 = _tmp[remaining_unpicked_items][chosen_point]
        assert remaining_unpicked_items[_tmp2] == 1
        remaining_unpicked_items[_tmp2] = 0
        assert remaining_unpicked_items[_tmp2] == 0

    assert (~remaining_unpicked_items).sum() \
        == config.num_points_to_label_per_al_iter

    return points_to_label


def get_labeled_and_topk_unlabeled_embeddings(config):
    """Return a tuple of (
        labeled training data embeddings,
        unlabeled training data embeddings for N highest entropy items,
        unlabeled index of the N high entropy items
    )
    The unlabeled index is an index over the unlabeled config._train_indices
    """
    # get model prediction on unlabeled points
    unlabeled_data_loader = feedforward.create_data_loader(
        config, idxs=config._train_indices[~config._is_labeled], shuffle=False)
    labeled_data_loader = feedforward.create_data_loader(
        config, idxs=config._train_indices[config._is_labeled], shuffle=False)

    # get labeled data embeddings
    embedding_labeled, _ = get_feature_embedding(
        config, labeled_data_loader, topk=None)
    # get unlabeled data embeddings on the N highest predictive entropy samples
    embedding_unlabeled, unlabeled_idxs = get_feature_embedding(
        config, unlabeled_data_loader, topk=config.num_max_entropy_samples)

    assert embedding_unlabeled.shape[0] \
        == unlabeled_idxs.shape[0]  # sanity check
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
            loader_idxs.extend(range(N, N+X.shape[0]))
            # select only top k values
            if topk is not None:
                _entropy = -yhat*torch.log2(yhat) - (1-yhat)*torch.log2(1-yhat)
                entropy = torch.cat([entropy, _entropy])
                if len(entropy) > topk:
                    entropy2, idxs = torch.topk(entropy, topk, dim=0)
                    embeddings = [embeddings[i] for i in idxs]
                    loader_idxs = [loader_idxs[i] for i in idxs]
                    entropy = entropy2
            N += X.shape[0]

        embeddings = torch.stack(embeddings)
        embeddings = embeddings.reshape(embeddings.shape[0], -1)
        loader_idxs = torch.tensor(
            loader_idxs, dtype=torch.long, device=config.device)
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

        # pick unlabeled points to label and label them
        if al_iter == 1:
            points_to_label = pick_initial_data_points_to_label(config)
        else:
            points_to_label = pick_data_points_to_label(config)

        # reset_model weights if necessary
        if config.reset_model_weights_each_al_iter:
            config.model.load_state_dict(
                pickle.loads(config._serialized_model_state_dict))

        # train model
        config.update_train_loader(points_to_label)
        feedforward.train(config)  # train for many epochs

        if config._is_labeled.sum() == config._is_labeled.shape[0]:
            print("Stop training.  Used up all available training data")
            break


class OnlineMedalMixin:
    reset_model_weights_each_al_iter = False

    # The percentage of previously labeled data points to include in training
    online_sample_frac = 0.0

    def update_train_loader(self, points_to_label):
        """This method is called inside the MedAL train loop train.
        Default settings are to train model using all labeled training data
        """
        if self.online_sample_frac is float:
            raise Exception("Must define online_sample_frac")

        # get a subset of the previously labeled points
        tmp = self._train_indices[self._is_labeled]
        if int(tmp.shape[0] * self.online_sample_frac) == 0:
            previously_labeled_points = torch.tensor(
                [], dtype=torch.long, device=self.device)
        else:
            tmpidxs = torch.randint(
                0, tmp.shape[0],
                (int(tmp.shape[0] * self.online_sample_frac), ))
            previously_labeled_points = tmp[tmpidxs]
        # get the newly labeled points
        _tmp = torch.arange(self._is_labeled.shape[0], device=self.device)
        newly_labeled_points = _tmp[~self._is_labeled][points_to_label]

        self._set_points_labeled(points_to_label)
        self.train_loader = feedforward.create_data_loader(
            self, idxs=torch.cat([
                previously_labeled_points, newly_labeled_points]))


class MedalConfigABC(feedforward.FeedForwardModelConfig):
    """Base class for all MedAL models"""
    run_id = str
    al_iters = int

    num_max_entropy_samples = int
    num_points_to_label_per_al_iter = int
    data_loader_num_workers = 0
    reset_model_weights_each_al_iter = True

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

    def _set_points_labeled(self, points_to_label):
        """Update self._is_labeled indices"""
        # label the unlabeled points.
        # --> unfortunately, have to do this in a convoluted way
        # since an update like arr[arr][idxs] = 1 doesn't actually update arr
        # if pytorch is using cuda.
        _test_sanity_check = self._is_labeled.sum()
        _tmp = torch.arange(self._is_labeled.shape[0], device=self.device)
        _tmp2 = _tmp[~self._is_labeled][points_to_label]
        # --> sanity check: point should not be previously labeled
        assert (self._is_labeled[_tmp2] == 0).all()
        self._is_labeled[_tmp2] = 1
        assert _test_sanity_check + len(points_to_label) \
            == self._is_labeled.sum()

    def update_train_loader(self, points_to_label):
        """
        Label the given unlabeled points and update self.train_loader to
        include these new points.
        This method is called from the MedAL train loop.
        The self.train_loader will include all labeled training data

        Subclasses could use points_to_label to train model in online fashion.
        Subclasses could use points_to_label to actually get a human labeler
        involved.
        """
        self._set_points_labeled(points_to_label)
        self.train_loader = feedforward.create_data_loader(
            self, idxs=self._train_indices[self._is_labeled])

    def __init__(self, config_override_dict):
        super().__init__(config_override_dict)

        # override the default feedforward config
        self.log_msg_minibatch = \
            "--> al_iter {config.cur_al_iter} " + self.log_msg_minibatch[4:]
        self.log_msg_epoch = \
            "al_iter {config.cur_al_iter} " + self.log_msg_epoch

        # split train set into unlabeled and labeled points
        self._train_indices = torch.tensor(
            self.train_loader.sampler.indices.copy(),
            dtype=torch.long, device=self.device)
        del self.train_loader  # will recreate this appropriately during train
        self._is_labeled = torch.ByteTensor(
            self._train_indices.shape).to(self.device) * 0

        self._serialized_model_state_dict = \
            pickle.dumps(self.model.state_dict())


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
    num_points_to_label_per_al_iter = 10

    def get_feature_embedding_layer(self):
        return list(self.model.children())[0][0][6]


class MedalResnet18BinaryClassifier(MedalConfigABC,
                                    BaselineResnet18BinaryClassifier):
    run_id = 'medal_resnet18'
    al_iters = 40

    num_max_entropy_samples = 50
    num_points_to_label_per_al_iter = 20

    def get_feature_embedding_layer(self):
        return list(self.model.children())[0][5]


class OnlineMedalResnet18BinaryClassifier(
        OnlineMedalMixin,
        MedalResnet18BinaryClassifier):
    pass
