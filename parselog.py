"""
Quick script to analyze Keras log files and make some plots.
"""
import sys
import re
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from os.path import join, exists

REGEXES_DATA_SHARED_ACROSS_ROWS = [
    r'^Epoch (?P<epoch>\d+)/\d+ *$',
    r'^Performing Active learning iteration (?P<al_iteration>\d+) for method ',
]
_REGEX_PERF = (
        r'^ *(?P<iteration>\d+)/\d+ \[[=\.>]+\] - .*'
        r'loss: (?P<train_loss>\d+\.\d+(e-?\d+)?) - '
        r'acc: (?P<train_acc>\d+.\d+(e-?\d+)?)'
     )
REGEXES_DATA_OF_A_ROW = [
    # given a line, try these regexes in sequence.  if a match is found, save
    # the data from the captured group as a row and stop evaluating the rest of
    # the regexes in the list.
    (
        _REGEX_PERF +
        r' - val_loss: (?P<val_loss>\d+.\d+(e-?\d+)?)'
        r' - val_acc: (?P<val_acc>\d+.\d+(e-?\d+)?) *$'
    ),
    _REGEX_PERF + ' *$',
]

SCHEMA = {
    'iteration': int,
    'train_loss': float,
    'train_acc': float,
    'epoch': int,
    'val_loss': float,
    'val_acc': float,
    'al_iteration': int,
    # TODO: train_set_size, oracle_set_size, true_pos_count, true_neg_count
}


def match(pat, line):
    m = re.match(pat, line)
    if m:
        return m.groupdict()
    return {}


def parse_log_files(fps_in):
    perf = []
    for fp_in in fps_in:
        with open(fp_in, 'r') as fin:
            dct = {}
            for line in fin:
                for pat in REGEXES_DATA_SHARED_ACROSS_ROWS:
                    dct.update(match(pat, line))
                dct2 = dct.copy()

                for pat in REGEXES_DATA_OF_A_ROW:
                    dct2 = match(pat, line)
                    if dct2:
                        dct2.update(dct.copy())
                        assert not set(dct2).difference(SCHEMA), \
                            "items missing in schema"
                        dct2 = {k: typ(dct2[k]) if k in dct2 else None
                                for k, typ in SCHEMA.items()}
                        perf.append(dct2)
                        break
    df = pd.DataFrame(perf)
    assert not df.empty, "Error parsing the log file.  Found no usable data."
    df['perf'] = ~df['val_acc'].isnull()
    if df['al_iteration'].isnull().all():
        df['al_iteration'] = df['al_iteration'].fillna(0)

    # sanity check
    # hack to ignore redundant log data when model stops and is resumed
    tmp = df.query('perf').groupby(['al_iteration'])['epoch'].count()
    bad_i = tmp.index[tmp > tmp.mode()[0]]
    if not bad_i.empty:
        print("--> Problem at al_iteration(s):  %s.  "
              "Found redundancy in log data (from re-running model and "
              "restarting an AL iteration). Ignoring earlier part of the "
              "redundant data, since it wasn't used to continue training."
              % bad_i.values)
        for i in bad_i.values:
            start_idx = df.query('al_iteration == @i').index[0]
            _tmp = df.query('al_iteration == @i')
            stop_idx = _tmp['epoch'].diff().idxmin() - 1
            df.drop(df.loc[start_idx:stop_idx].index, inplace=True)
    tmp = df.query('perf').groupby(['al_iteration'])['epoch'].count()
    if tmp.shape[0] > 1:
        assert 0 == tmp.var(), "bug trying to ignore redundant data"

    # hack: ignore end of log data if it's incomplete
    if df['al_iteration'].unique().shape[0] > 1:
        if (df.groupby('al_iteration')['epoch']
                .count().diff().tail(1) < 1).all():
            print("--> Dropping last al_iteration, since it was incomplete")
            _last_iter = df['al_iteration'].max()
            df.drop(df[df['al_iteration'] == _last_iter].index, inplace=True)

    # sanity check data:
    assert (df['val_acc'].isnull() == df['val_loss'].isnull()).all()
    each_epoch_of_an_al_iteration_is_completed = \
        df.groupby(['al_iteration', 'epoch'])['iteration']\
        .agg(lambda x: x.count() == x.count().max()).all()
    if not each_epoch_of_an_al_iteration_is_completed:
        print(
            "\n\nWARNING: \n Run may not have completed successfully."
            " At least one epoch does not have enough iterations\n")
    return df


def plot_learning_curve_over_al_iterations(
        df, col_suffix, last_al_iteration, selected_al_iters):
    """col_suffix is either "loss" or "acc" """
    f, (ax1, ax2) = plt.subplots(2, 1)

    cols = ['train_%s' % col_suffix, 'val_%s' % col_suffix]
    #  data = df.query('perf').query('al_iteration in @selected_al_iters')

    def _plot_learning_curve(*args, **kws):
        data = kws.pop('data')
        data[cols].plot(style='-', linewidth=.5, use_index=False, ax=plt.gca())

    if last_al_iteration > 0:
        f = sns.FacetGrid(
            df.query('perf'),
            col='al_iteration', col_wrap=np.round(np.sqrt(last_al_iteration)))\
            .map_dataframe(_plot_learning_curve).add_legend().fig
    else:
        f = df.query('perf').query('al_iteration == @last_al_iteration')[cols]\
            .plot(title="for AL Iteration %s" % (last_al_iteration+1)).figure

    # plot train and val  loss|acc for every epoch of given al iteration.
    # only show some al iterations
    #  data[cols].plot(
    #          title="for %s of %s AL iterations" % (
    #              len(selected_al_iters), last_al_iteration+1),
    #          style='-', linewidth=.5, use_index=False, ax=ax1)
    #  ax1.vlines(
    #      np.bincount(data['al_iteration'].values).cumsum(),
    #      *ax1.get_ylim(), linestyle='--', alpha=.5)

    #  # plot only the last al iteration
    #  df.query('perf').query('al_iteration == @last_al_iteration')[cols].plot(
    #      title="for AL Iteration %s" % (last_al_iteration+1), ax=ax2)

    f.suptitle("%s vs Epoch" % col_suffix.capitalize())
    f.subplots_adjust(top=.9)
    #  f.tight_layout(rect=[0, 0.03, 1, 0.95])
    f.savefig(join(img_dir, "%s_vs_epoch.png" % col_suffix))


def plot_heatmap(df, values_col, values_col_name_for_title, selected_al_iters):
    def make_heatmap(values_col, *args, **kwargs):
        dat = kwargs.pop('data')
        d = dat.pivot('iteration', 'epoch', values_col)
        sns.heatmap(d, *args, **kwargs)

    f = sns.FacetGrid(
        df.query('al_iteration in @selected_al_iters'),
        row='al_iteration', dropna=False, margin_titles=True, aspect=4,
        sharey=False)\
        .map_dataframe(make_heatmap, values_col, cbar=False)

    f.fig.suptitle(
        "%s as we vary Epoch and Iteration" % values_col_name_for_title)
    f.set_axis_labels("Epoch", "Iteration")
    #  f.set_ylabels("Iteration")
    f.fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    f.savefig(join(img_dir, "iteration_vs_epoch_%s.png" % values_col))


def plot_heatmap_at_al_iter(df, al_iteration):
    df = df.query('al_iteration == @al_iteration')
    f, (ax1, ax2) = plt.subplots(2, 1)
    f.suptitle("Epoch vs Iteration on AL Iteration %s: Train Performance"
               % (al_iteration+1))
    ax1.set_title("Training Accuracy")
    sns.heatmap(df.pivot('iteration', 'epoch', 'train_acc'), ax=ax1)
    ax2.set_title("Training Loss")
    sns.heatmap(df.pivot('iteration', 'epoch', 'train_loss'), ax=ax2)
    f.tight_layout(rect=[0, 0.03, 1, 0.95])
    f.savefig(join(img_dir,
                   "iteration_vs_epoch_al_iter_%s.png" % (al_iteration+1)))


def quantile_perf_across_al_iterations(df):
    al_perf = df.groupby('al_iteration')[
        ['val_acc', 'val_loss', 'train_acc', 'train_loss']]\
        .quantile([0, .1, .25, .5, .75, .9, 1]).unstack()
    al_perf.columns.names = ('variable', 'quantile')
    for cols, name in [(['train_loss', 'val_loss'], "Loss"),
                       (['train_acc', 'val_acc'], "Accuracy")]:
        f, axs = plt.subplots(3, 2)
        for quantile, axss in zip([.1, .5, .9], axs):
            al_perf[(cols[0], quantile)].plot(
                title="Quantile: %s, %s" % (quantile, cols[0]), ax=axss[0])
            al_perf[(cols[1], quantile)].plot(
                title="Quantile: %s, %s" % (quantile, cols[1]), ax=axss[1])

        f.suptitle("Quantile %s of model across AL iterations" % name)
        f.tight_layout(rect=[0, 0.03, 1, 0.95])
        f.savefig(join(
            img_dir, "quantile_%s_across_al_iter.png" % name.lower()))


if __name__ == "__main__":
    img_dir = sys.argv[1]
    fps_in = sys.argv[2:]
    assert exists(img_dir)

    print("Analyzing log:  %s" % fps_in)
    print("Saving images to directory:  %s" % img_dir)

    df = parse_log_files(fps_in)

    last_al_iter = df['al_iteration'].max()
    selected_al_iters = np.unique(np.linspace(0, last_al_iter, 6, dtype='int'))

    print("Generating several plots...")

    # learning curves
    plot_learning_curve_over_al_iterations(
        df, 'loss', last_al_iter, selected_al_iters)
    plot_learning_curve_over_al_iterations(
        df, 'acc', last_al_iter, selected_al_iters)

    # how learning progresses over time.
    plot_heatmap(df, 'train_acc', "Train Accuracy", selected_al_iters)
    plot_heatmap(df, 'train_loss', "Train Loss", selected_al_iters)
    plot_heatmap_at_al_iter(df, last_al_iter)

    # AL performance over time
    if last_al_iter > 0:
        quantile_perf_across_al_iterations(df)
