"""
Analyze Keras and MedAL Pytorch log files and make some plots.
"""
import re
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from os.path import join
import os
import argparse as ap
import abc


class LogType(abc.ABC):
    # A list of regexes.
    regexes_data_shared_across_rows = NotImplemented  # type: List[regex]

    # A list of regexes.  Given a line of log data, try these regexes in
    # sequence.  If a match is found, save the data from the captured group as
    # a row and stop evaluating the rest of the regexes in the list.
    regexes_data_of_a_row = NotImplemented  # type: List[regex]


class KerasConfig(LogType):
    regexes_data_shared_across_rows = [
        r'^Epoch (?P<epoch>\d+)/\d+ *$',
        (r'^Performing Active learning iteration '
         r'(?P<al_iter>\d+) for method '),
    ]
    _regex_perf = (
            r'^ *(?P<batch_idx>\d+)/\d+ \[[=\.>]+\] - .*'
            r'loss: (?P<train_loss>\d+\.\d+(e-?\d+)?) - '
            r'acc: (?P<train_acc>\d+.\d+(e-?\d+)?)'
        )
    regexes_data_of_a_row = [
        (
            _regex_perf +
            r' - val_loss: (?P<val_loss>\d+.\d+(e-?\d+)?)'
            r' - val_acc: (?P<val_acc>\d+.\d+(e-?\d+)?) *$'
        ),
        _regex_perf + ' *$',
    ]


class MedALConfig(LogType):
    _epoch = r"epoch:?\s+(?P<epoch>\d+)\s+"
    _batch_idx = r"batch_idx:?\s+(?P<batch_idx>\d+)\s+"
    _train_loss = r"train_loss:?\s+(?P<train_loss>\d+\.\d+(e-?\d+)?)\s+"
    _val_loss = r"val_loss:?\s+(?P<val_loss>\d+\.\d+(e-?\d+)?)\s+"
    _train_acc = r"train_acc:?\s+(?P<train_acc>\d+\.\d+(e-?\d+)?)\s*"
    _val_acc = r"val_acc:?\s+(?P<val_acc>\d+\.\d+(e-?\d+)?)\s*"
    _al_iter = r"(al_iter:?\s+(?P<al_iter>\d+)\s*)?"

    regexes_data_shared_across_rows = [
    ]

    regexes_data_of_a_row = [
        # given a line, try these regexes in sequence.  if a match is found,
        # save the data from the captured group as a row and stop evaluating
        # the rest of the regexes in the list.
        (r'^' + _al_iter + _epoch + _train_loss + _val_loss + _train_acc +
         _val_acc + '$'),
        (r'^-->\s+' + _al_iter + _epoch + _batch_idx + _train_loss + _train_acc
         + '$'),
    ]


class Optional:
    def __init__(self, typ):
        self.typ = typ

    def __call__(self, x):
        if x is not None:
            return self.typ(x)


SCHEMA = {
    'batch_idx': int,
    'train_loss': float,
    'train_acc': float,
    'epoch': int,
    'val_loss': float,
    'val_acc': float,
    'al_iter': Optional(int),
    # TODO: train_set_size, oracle_set_size, true_pos_count, true_neg_count
}


def match(pat, line):
    m = re.match(pat, line)
    if m:
        return m.groupdict()
    return {}


def _parse_log_files_to_df(log_type, fps_in):
    perf = []
    for fp_in in fps_in:
        with open(fp_in, 'r') as fin:
            dct = {}
            for line in fin:
                for pat in log_type.regexes_data_shared_across_rows:
                    dct.update(match(pat, line))
                dct2 = dct.copy()

                for pat in log_type.regexes_data_of_a_row:
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
    return df


def _parse_log_sanitize_and_clean(df):
    df = df.copy()
    assert not df.empty, "Error parsing the log file.  Found no usable data."
    df['perf'] = ~df['val_acc'].isnull()
    if df['al_iter'].isnull().all():
        df['al_iter'] = df['al_iter'].fillna(0)

    # sanity check
    # hack to ignore redundant log data when model stops and is resumed
    tmp = df.query('perf').groupby(['al_iter'])['epoch'].count()
    bad_i = tmp.index[tmp > tmp.mode()[0]]
    if not bad_i.empty:
        print("--> Problem at al_iter(s):  %s.  "
              "Found redundancy in log data (from re-running model and "
              "restarting an AL iter). Ignoring earlier part of the "
              "redundant data, since it wasn't used to continue training."
              % bad_i.values)
        for i in bad_i.values:
            start_idx = df.query('al_iter == @i').index[0]
            _tmp = df.query('al_iter == @i')
            stop_idx = _tmp['epoch'].diff().idxmin() - 1
            df.drop(df.loc[start_idx:stop_idx].index, inplace=True)

    # hack: ignore end of log data if it's incomplete
    if df['al_iter'].unique().shape[0] > 1:
        if (df.groupby('al_iter')['epoch']
                .count().diff().tail(1) < 1).all():
            print("--> Dropping last al_iter, since it was incomplete")
            _last_iter = df['al_iter'].max()
            df.drop(df[df['al_iter'] == _last_iter].index, inplace=True)

    # sanity check
    tmp = df.query('perf').groupby(['al_iter'])['epoch'].count()
    if tmp.shape[0] > 1 and 0 != tmp.var():
        print("WARNING: Varying rows of data per al iter.  Unless early"
              " stopping was applied to this model, something may be wrong.")

    # sanity check data:
    assert (df['val_acc'].isnull() == df['val_loss'].isnull()).all()
    each_epoch_of_an_al_iter_is_completed = \
        df.groupby(['al_iter', 'epoch'])['batch_idx']\
        .agg(lambda x: x.count() == x.count().max()).all()
    if not each_epoch_of_an_al_iter_is_completed:
        print(
            "\n\nWARNING: \n Run may not have completed successfully."
            " At least one epoch does not have enough batch_idxs\n")
    return df


def _parse_log_files(fps_in, log_type):
    df = _parse_log_files_to_df(log_type, fps_in)
    df = _parse_log_sanitize_and_clean(df)
    return df


def parse_log_files(fps_in, log_type=None):
    if log_type is not None:
        return _parse_log_files(fps_in, log_type)

    for log_type in [MedALConfig]:  # , KerasConfig]:
        df = None
        try:
            df = _parse_log_files(fps_in, log_type)
            print('--', log_type.__name__, "successfully parsed log file")
            assert not df.empty
            break
        except Exception as e:
            print('--', log_type.__name__, e)
            raise
    if df is None:
        raise Exception("Could not figure out how to parse the given log"
                        " file(s): %s" % (str(config.fps_in)))
    return df


def plot_learning_curve_over_al_iters(
        img_dir, df, col_suffix, last_al_iter, selected_al_iters):
    """col_suffix is either "loss" or "acc" """
    f, (ax1, ax2) = plt.subplots(2, 1)

    cols = ['train_%s' % col_suffix, 'val_%s' % col_suffix]
    #  data = df.query('perf').query('al_iter in @selected_al_iters')

    def _plot_learning_curve(*args, **kws):
        data = kws.pop('data')
        data[cols].plot(
            style='-', grid=True, linewidth=.5, use_index=False, ax=plt.gca())

    if last_al_iter > 0:
        f = sns.FacetGrid(
            df.query('perf').drop('batch_idx', axis=1).set_index('epoch'),
            col='al_iter', col_wrap=np.round(np.sqrt(last_al_iter)))\
            .map_dataframe(_plot_learning_curve).add_legend().fig
        f.subplots_adjust(top=.85)
    else:
        ax = df.query('perf')\
            .set_index('epoch')\
            .query('al_iter == @last_al_iter')[cols]\
            .plot(grid=True)
        table = pd.plotting.table(
            ax, df.query('perf')[cols].describe().round(4).loc[['max', 'min']],
            loc='lower center', colWidths=[0.2, 0.2, 0.2], alpha=.4)
        table.auto_set_font_size(False)
        f = ax.figure

    # plot train and val  loss|acc for every epoch of given al iter.
    # only show some al iters
    #  data[cols].plot(
    #          title="for %s of %s AL iters" % (
    #              len(selected_al_iters), last_al_iter+1),
    #          style='-', linewidth=.5, use_index=False, ax=ax1)
    #  ax1.vlines(
    #      np.bincount(data['al_iter'].values).cumsum(),
    #      *ax1.get_ylim(), linestyle='--', alpha=.5)

    #  # plot only the last al iter
    #  df.query('perf').query('al_iter == @last_al_iter')[cols].plot(
    #      title="for AL Iter %s" % (last_al_iter+1), ax=ax2)

    f.suptitle("%s vs Epoch" % col_suffix.capitalize())
    #  f.tight_layout(rect=[0, 0.03, 1, 0.95])
    f.savefig(join(img_dir, "%s_vs_epoch.png" % col_suffix))


def plot_heatmap(img_dir, df, values_col, values_col_name_for_title,
                 selected_al_iters):
    def make_heatmap(values_col, *args, **kwargs):
        dat = kwargs.pop('data')
        d = dat.pivot('batch_idx', 'epoch', values_col)
        sns.heatmap(d, *args, **kwargs)

    f = sns.FacetGrid(
        df.query('not perf and al_iter in @selected_al_iters'),
        row='al_iter', dropna=False, margin_titles=True, aspect=4,
        sharey=False)\
        .map_dataframe(make_heatmap, values_col, cbar=False)

    f.fig.suptitle(
        "%s as we vary Epoch and Batch index" % values_col_name_for_title)
    f.set_axis_labels("Epoch", "Batch index")
    #  f.set_ylabels("Iter")
    f.fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    f.savefig(join(img_dir, "batch_idx_vs_epoch_%s.png" % values_col))


def plot_heatmap_at_al_iter(img_dir, df, al_iter):
    df = df.query('not perf and al_iter == @al_iter')
    f, (ax1, ax2) = plt.subplots(2, 1)
    f.suptitle(
        "Epoch vs Batch%s: Train Performance" %
        ("on AL Iter" % (al_iter+1) if df['al_iter'].max() > 0 else ""))
    ax1.set_title("Training Accuracy")
    sns.heatmap(df.pivot('batch_idx', 'epoch', 'train_acc'), ax=ax1)
    ax2.set_title("Training Loss")
    sns.heatmap(df.pivot('batch_idx', 'epoch', 'train_loss'), ax=ax2)
    f.tight_layout(rect=[0, 0.03, 1, 0.95])
    f.savefig(join(img_dir,
                   "batch_idx_vs_epoch_al_iter_%s.png" % (al_iter+1)))


def plot_quantile_perf_across_al_iters(img_dir, df):
    # TODO: these plots can be deceptive since at each AL iter, there are
    # more batch_idxs and therefore the probability of a higher .9 and a lower
    # .1 increases as AL iter increases.  maybe can normalize for this
    # probability?
    al_perf = df.query('perf').groupby('al_iter')[
        ['val_acc', 'val_loss', 'train_acc', 'train_loss']]\
        .quantile([0, .01, .25, .5, .75, .99, 1]).unstack()
    al_perf.columns.names = ('variable', 'quantile')
    for cols, name in [(['train_loss', 'val_loss'], "Loss"),
                       (['train_acc', 'val_acc'], "Accuracy")]:
        f, axs = plt.subplots(3, 2)
        for quantile, axss in zip([.01, .5, .99], axs):
            al_perf[(cols[0], quantile)].plot(
                title="Quantile: %s, %s" % (quantile, cols[0]), ax=axss[0],
                grid=True)
            al_perf[(cols[1], quantile)].plot(
                title="Quantile: %s, %s" % (quantile, cols[1]), ax=axss[1],
                grid=True)

        f.suptitle("Quantile %s of model across AL iters" % name)
        f.tight_layout(rect=[0, 0.03, 1, 0.95])
        f.savefig(join(
            img_dir, "quantile_%s_across_al_iter.png" % name.lower()))


def build_arg_parser():
    p = ap.ArgumentParser()
    p.add_argument(
        "output_dir", help="where to write results")
    p.add_argument(
        "fps_in", nargs='+', help="log files representing one training run")
    return p


if __name__ == "__main__":
    config = build_arg_parser().parse_args()
    print(config)
    print("Analyzing log:  %s" % config.fps_in)
    print("Saving images to directory:  %s" % config.output_dir)

    df = parse_log_files(config.fps_in)

    os.makedirs(config.output_dir, exist_ok=True)
    df.to_csv(join(config.output_dir, "logdata.csv"), index=False)

    print("Generating several plots...")

    last_al_iter = df['al_iter'].max()
    selected_al_iters = np.unique(np.linspace(0, last_al_iter, 6, dtype='int'))

    # learning curves
    plot_learning_curve_over_al_iters(
        config.output_dir, df, 'loss', last_al_iter, selected_al_iters)
    plot_learning_curve_over_al_iters(
        config.output_dir, df, 'acc', last_al_iter, selected_al_iters)

    # how learning progresses over time.
    # not relevant for online medal, since there may be no batch indexes
    if not df.query('not perf').empty:
        plot_heatmap(
            config.output_dir, df, 'train_acc', "Train Accuracy",
            selected_al_iters)
        plot_heatmap(
            config.output_dir, df, 'train_loss', "Train Loss",
            selected_al_iters)
        plot_heatmap_at_al_iter(config.output_dir, df, last_al_iter)

    # AL performance over time
    if last_al_iter > 0:
        plot_quantile_perf_across_al_iters(config.output_dir, df)
