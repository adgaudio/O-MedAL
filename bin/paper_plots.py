from matplotlib import pyplot as plt
import matplotlib.ticker as mticker
from matplotlib import patches
import matplotlib
SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 14
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
from os.path import join
import pandas as pd
import seaborn as sns
import re
import numpy as np

# define log csv data and dataset parameters not available in csvs
analysis_dir = './data/analysis'  # dir to save plots to
points_to_label_per_al_iter = 20
train_set_size = 949

fp_baseline = "data/_analysis/R6-20190315T030959.log/logdata.csv"
#  fp_medal = "data/_analysis/RM6-20190319T005512.log/logdata.csv"
fp_medal_patience10 = "data/_analysis/RM6e-20190324T140934.log/logdata.csv"
fp_medal_patience20 = "data/_analysis/RM6g-20190406T222759.log/logdata.csv"
fps_varying_online_frac = [
    "data/_analysis/RMO6-0d-20190323T003224.log/logdata.csv",
    "data/_analysis/RMO6-12.5d-20190322T142710.log/logdata.csv",
    "data/_analysis/RMO6-25d-20190323T152428.log/logdata.csv",
    "data/_analysis/RMO6-37.5d-20190322T051602.log/logdata.csv",
    "data/_analysis/RMO6-50d-20190323T030509.log/logdata.csv",
    "data/_analysis/RMO6-62.5d-20190322T092309.log/logdata.csv",
    "data/_analysis/RMO6-75d-20190323T185418.log/logdata.csv",
    "data/_analysis/RMO6-87.5d-20190322T173655.log/logdata.csv",
    "data/_analysis/RMO6-100d-20190323T082726.log/logdata.csv",
]


def get_train_frac(fp):
    return str(float(re.sub(r'.*RMO6-([\d\.]+).?-.*', r'\1', fp)) * .01)


# load the data
dfb = pd.read_csv(fp_baseline).query('perf').sort_values('epoch').set_index('epoch')
dfm20 = pd.read_csv(fp_medal_patience20).query('perf').sort_values(['al_iter', 'epoch'])
dfm10 = pd.read_csv(fp_medal_patience10).query('perf').sort_values(['al_iter', 'epoch'])
dfs = {'Online - ' + get_train_frac(fp): pd.read_csv(fp).query('perf')
       for fp in fps_varying_online_frac}
dfo = pd.concat(dfs, names=['Experiment', 'log_line_num'])
dfo['online_sample_frac'] = dfo.index.get_level_values('Experiment')\
    .str.extract('Online - (.*)').astype('float').values
dfo = dfo.sort_values(['online_sample_frac', 'al_iter', 'epoch'])
# --> reindex dfm20 and dfo to guarantee if the run was incomplete we show empty
# space in plot that would represent usage of the full dataset.
_mi = pd.MultiIndex.from_product([
        np.arange(1, 49), np.arange(1, 151)], names=['al_iter', 'epoch'])
#  dfm20 = dfm20.set_index(['al_iter', 'epoch']).reindex(_mi).reset_index()
assert dfm20.dropna(axis=1, how='all').dropna().groupby('al_iter').count()['epoch'].min() >= dfo['epoch'].max()
assert dfm10.dropna(axis=1, how='all').dropna().groupby('al_iter').count()['epoch'].min() >= dfo['epoch'].max()

# compute percent data labeled (for x axis of plot)
for df in [dfo, dfm10, dfm20]:
    N = df['al_iter'].values * points_to_label_per_al_iter
    df['pct_dataset_labeled_int'] = (N / train_set_size * 100).astype(int)
    x = (N / train_set_size * 100)
    #  x[x > 100] = 100  # clip ends, since the points_to_label_per_al_iter is not divisble by train_set_size and since the reindexing operation would make seem like over 100%
    df['pct_dataset_labeled'] = x

# compute num examples processed.  medal and omedal have +1 because initial
# train set is 1 + points_to_label_per_al_iter.
dfb['num_img_patches_processed'] = dfb.index * train_set_size
dfm10['num_img_patches_processed'] = \
    (dfm10['al_iter'] * points_to_label_per_al_iter + 1).cumsum()
dfm20['num_img_patches_processed'] = \
    (dfm20['al_iter'] * points_to_label_per_al_iter + 1).cumsum()
dfo['num_img_patches_processed'] = \
    (points_to_label_per_al_iter
     + np.floor(
         dfo['online_sample_frac']
         * (1 + (dfo['al_iter']-1) * points_to_label_per_al_iter)))\
    .unstack('Experiment').cumsum().stack('Experiment')\
    .swaplevel().sort_index()

# get one row per experiment per al iter.
Z = dfo\
    .groupby(['online_sample_frac', 'al_iter'])\
    .agg({'pct_dataset_labeled': 'first',
          'num_img_patches_processed': 'max',
          'val_acc': 'max'})\
    .reset_index()
baseline_num_processed = dfb.query('epoch <= 80')['num_img_patches_processed'].max()
baseline_max_acc = dfb.query('epoch <= 80')['val_acc'].max()  # max with about 20 extra epochs after convergence
Z['process_more_pts_than_baseline'] = Z['num_img_patches_processed'] > baseline_num_processed
Z['val_acc_worse_than_baseline'] = Z['val_acc'] < baseline_max_acc


# get keypoints for next couple plots
_tmp = dfo[['val_acc', 'pct_dataset_labeled', 'num_img_patches_processed']]
keypoints = [
    # online experiments
    (_tmp.loc[dfo['val_acc'].idxmax()], 'dimgray'),
    #  (_tmp.loc[dfo.query('online_sample_frac == 0.375')['val_acc'].idxmax()],
     #  'red'),
    (_tmp.loc[dfo[dfo['val_acc'] >= baseline_max_acc]['pct_dataset_labeled'].idxmin()],
    #  (_tmp.loc[dfo.query('online_sample_frac == 0.875')
             #  .sort_values('val_acc', ascending=False)
             #  .head(5)['pct_dataset_labeled'].idxmin()],
     'silver'),
    (_tmp.loc[ _tmp.loc[_tmp['val_acc'] >= baseline_max_acc]['num_img_patches_processed'].idxmin()],
     'darkorange'),  # TODO
    # medal
    (dfm20.loc[dfm20['val_acc'].idxmax()].loc[['val_acc', 'pct_dataset_labeled', 'num_img_patches_processed']]\
     .rename(('MedAL (patience=20)', '')), 'black'),
    (dfm10.loc[dfm10['val_acc'].idxmax()].loc[['val_acc', 'pct_dataset_labeled', 'num_img_patches_processed']]\
     .rename(('MedAL (patience=10)', '')), 'darkblue'),
]

# plot 1: val acc vs percent dataset labeled
# --> prepare results for plot
def main_perf_plot(subset_experiments=(), add_medal_to_legend=False, add_baseline_to_legend=False):
    medalpltdata = dfm20\
        .set_index(['pct_dataset_labeled_int', 'epoch'])['val_acc']\
        .rename('MedAL (patience=20)')
    onlinepltdata = dfo\
        .query('perf')\
        .set_index(['pct_dataset_labeled_int', 'epoch'], append=True)\
        .drop('al_iter', axis=1)\
        .droplevel('log_line_num')['val_acc']\
        .unstack('Experiment')\
        .reindex(columns=sorted(dfo.index.levels[0], key=lambda x: float(x.replace('Online - ', ''))))\
        .reindex(medalpltdata.index)
    if subset_experiments:
        onlinepltdata = onlinepltdata[subset_experiments]
    # --> add the plots
    if len(subset_experiments) == 1:
        fig, ax = plt.subplots(1, 1, figsize=(8, 2))
    else:
        fig, ax = plt.subplots(1, 1, figsize=(8, 11))
    axs = onlinepltdata\
        .plot(ylim=(0, 1), color='red', subplots=True, ax=ax)
    if add_medal_to_legend:
        [medalpltdata.plot(ax=ax, alpha=.6, color='green', linewidth=0.4)
         for ax in axs]
    else:
        [medalpltdata.plot(ax=ax, alpha=.6, color='green', linewidth=0.4,
                           label='_nolegend_')
         for ax in axs]
    if add_baseline_to_legend:
        [ax.hlines(baseline_max_acc, 0, medalpltdata.shape[0],
                color='dodgerblue', linestyle='--', label='Baseline ResNet18')
         for ax in axs]
    else:
        [ax.hlines(baseline_max_acc, 0, medalpltdata.shape[0],
                color='dodgerblue', linestyle='--', label='_nolegend_')
         for ax in axs]
    [ax.legend(loc='lower left', frameon=True, ncol=3)
     for ax in axs]
    # --> handle xticks.
    [ax.set_xlabel('Percent Dataset Labeled') for ax in axs]
    #  axs[-1].set_xlabel('Percent Dataset Labeled')
    [ax.set_ylabel('Test Accuracy') for ax in axs]
    #  axs[len(axs)//2].set_ylabel('Test Accuracy')
    axs[0].xaxis.set_major_locator(mticker.LinearLocator(10))
    axs[0].xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, pos: min(100, medalpltdata.index[int(x)][0])))
    fig.tight_layout()
    fig.savefig(join(analysis_dir, 'varying_online_frac%s.png'
                               % len(subset_experiments)))
main_perf_plot()
main_perf_plot([
    'Online - 0.0', 'Online - 0.125', 'Online - 0.875', 'Online - 1.0'],
    add_medal_to_legend=True, add_baseline_to_legend=True)
main_perf_plot(['Online - 0.875'], add_medal_to_legend=True)
main_perf_plot(['Online - 0.875', 'Online - 0.125'], add_medal_to_legend=True)

# plot 2: training time (number of image patches used)
# --> plot amount of data used as we change the sample frac
def plot_training_time(logy=True, fracs=None, use_keypoints=True):
    f, ax = plt.subplots(1, 1, figsize=(6, 4))
    tmp = Z.set_index(['pct_dataset_labeled', 'online_sample_frac'])\
        ['num_img_patches_processed']\
        .unstack('online_sample_frac')\
        .join(dfm20.groupby('pct_dataset_labeled')['num_img_patches_processed']
              .max().rename('MedAL (patience=20)'), how='outer')\
        .join(dfm10.groupby('pct_dataset_labeled')['num_img_patches_processed']
              .max().rename('MedAL (patience=10)'), how='outer')
    # plot exponential curves for experiments and MedAL
    tmp\
        .drop('MedAL (patience=20)', axis=1)\
        .drop('MedAL (patience=10)', axis=1)\
        .plot(ax=ax, logy=logy, legend=False)
    tmp['MedAL (patience=20)'].plot(ax=ax, style='-.', logy=logy, color='black', legend=False)
    tmp['MedAL (patience=10)'].plot(ax=ax, style=':', logy=logy, color='darkblue', legend=False)

    # plot baseline horizontal line
    ax.hlines(baseline_num_processed, 0, 100,
              color='dodgerblue', linestyle='--', label='Baseline ResNet')

    # plot dots for highest val acc above and below baseline line.
    if fracs == 'all':
        tmp2 = Z.copy()
    elif fracs:
        tmp2 = Z.query('online_sample_frac in @fracs').copy()
    else:
        tmp2 = None
        points = ()
    if tmp2 is not None:
        points = Z.loc[tmp2.groupby(['online_sample_frac', 'process_more_pts_than_baseline'])['val_acc'].idxmax()]
        # add model with the lowest num data processed that still outperforms
        # baseline
        points = points.append(Z.loc[ Z.loc[Z['val_acc'] >= baseline_max_acc]['num_img_patches_processed'].idxmin()])

        #  add medal to points
        points = points.append(dfm20.loc[dfm20['val_acc'].idxmax()])
        points = points.append(dfm10.loc[dfm10['val_acc'].idxmax()])

        def norm(arr, n=50):
            arr = arr - arr.min()/n*(n-1)
            return arr / arr.max()
        if len(points) == 5: # hack
            color = ['red', 'dimgray', 'dimgray', 'red', 'black']
        else:
            color = 'black' #'black'  # plt.cm.spring_r(norm(points['val_acc'].values, 50))
        points.plot.scatter(
            'pct_dataset_labeled', 'num_img_patches_processed', marker='*',
            ax=ax, c=color, s=norm(points['val_acc'].values)*200
        )
        points.iloc[[-1]].plot.scatter(
            'pct_dataset_labeled', 'num_img_patches_processed', marker='*',
            ax=ax, c='black', s=(norm(points['val_acc'].values)*200)[-1]
        )
    if use_keypoints:
        [ax.plot(xy[1], xy[2], marker, markersize=ms,
                 color=color if color2 is None else color2, alpha=1)
         for xy, color in keypoints
         for marker, ms, color2 in [('P', 25, None), ('+', 35, 'white')]]

    ax.set_ylabel('Number of Examples Processed%s'
                  % (' (log scale)' if logy else ''))
    ax.set_xlabel('Percent Dataset Labeled')
    ax.legend(ncol=3)

    cols = ['pct_dataset_labeled', 'num_img_patches_processed']
    rows = 'online_sample_frac'

    f.savefig(join(
        analysis_dir,
        'num_img_patches_processed%s%s%s.png'
        % (("_logy" if logy else ""), len(points),
           '_kp' if use_keypoints else '')))

plot_training_time(logy=False, fracs=None, use_keypoints=False)
plot_training_time(logy=True, fracs=[], use_keypoints=False)
plot_training_time(logy=True, fracs=[], use_keypoints=False)
plot_training_time(logy=True, fracs=[], use_keypoints=True)
plot_training_time(logy=True, fracs='all', use_keypoints=False)
plot_training_time(logy=True, fracs=['0.875', '0.125'], use_keypoints=False)

# latex table of val accs above and below baseline.
with open(join(analysis_dir, 'table.tex'), 'w') as fout:
    points = Z.loc[Z.groupby([
        'online_sample_frac', 'process_more_pts_than_baseline'
    ])['val_acc'].idxmax()]
    table_data = points.query('not val_acc_worse_than_baseline')\
        .set_index('online_sample_frac')\
        .sort_values(['num_img_patches_processed', 'val_acc'])[
            ['val_acc', 'pct_dataset_labeled', 'num_img_patches_processed',
             'process_more_pts_than_baseline',
             'val_acc_worse_than_baseline']]
    fout.write(table_data.to_latex())
    print(table_data.to_string())


# plot 3: best performing model
topn = 10
g = dfo.groupby('Experiment')\
    .apply(lambda x: x.sort_values(['val_acc', 'pct_dataset_labeled'],
                                   ascending=False).head(topn))\
    .sort_values('online_sample_frac')\
    [['val_acc', 'pct_dataset_labeled', 'al_iter', 'online_sample_frac']]\
    .droplevel(0).droplevel('log_line_num').reset_index()
f, ax = plt.subplots(1, 1, figsize=(6, 4))
# --> add jitter
x_jitter = g['val_acc'].value_counts().index.to_series()\
    .sort_values().diff().min() / 6
y_jitter = g['pct_dataset_labeled'].value_counts().index.to_series()\
    .sort_values().diff().min() / 6
g['val_acc'] += x_jitter * (np.random.randint(-1, 1, g.shape[0])*2+1)
g['pct_dataset_labeled'] += \
    y_jitter * (np.random.randint(-1, 1, g.shape[0])*2+1)
# --> make scatter plot
sns.scatterplot(
    'Percent Dataset Labeled', 'Test Accuracy', hue='Experiment',
    data=g.rename({
        'pct_dataset_labeled': 'Percent Dataset Labeled',
        'val_acc': 'Test Accuracy', }, axis=1), ax=ax,
    #  palette=sns.palplot(sns.color_palette("cubehelix", 9))
    #  palette=sns.palplot(sns.color_palette("coolwarm", 9))
    palette=sns.palplot(sns.color_palette("hsv", 9)),
    #  legend='brief'
    #  palette='GnBu_d')
)
ax.hlines(baseline_max_acc, 0, 100, linestyle='--',
          color='dodgerblue', alpha=1, label='ResNet18 (best accuracy)')
ax.hlines(dfm20['val_acc'].max(), 0, 100, linestyle='-.', alpha=1,
          color='black', label='MedAL, patience=20\n    (best accuracy)')
ax.hlines(dfm10['val_acc'].max(), 0, 100, linestyle=':', alpha=1,
          color='darkblue', label='MedAL, patience=10\n    (best accuracy)')
ax.legend(framealpha=.8, frameon=True, bbox_to_anchor=(0, .34), loc='center left', ncol=2)
ax.set_title("Top %s Highest Test Accuracies For Each Experiment" % topn)
# add annotations to key points of interest on plot
with open(join(analysis_dir, 'topn_keypoint_table.tex'), 'w') as fout:
    print("\n\n  ---  keypoint table  ---")
    d = pd.DataFrame([x[0] for x in keypoints])
    d.index = pd.MultiIndex.from_tuples(d.index)
    d = d.reset_index(level=1, drop=True)
    d.index.name = 'Experiment'
    d = d.append(pd.Series([baseline_max_acc, 100, baseline_num_processed],
                       index=d.columns, name='ResNet18 Baseline'))

    d2 = pd.DataFrame(index=d.index)
    def to_float2(num):
        return '%0.2f' % num
    def to_pfloat2(num):
        return '%+0.2f' % num
    def to_int(num):
        return '%i' % num

    d2['Test Accuracy'] = (
        (d['val_acc'] * 100).apply(to_float2)
        + r'\% (' +
        ((d['val_acc']-baseline_max_acc)*100).apply(to_pfloat2)
        + r'\%)')
    d2['Percent Labeled'] = d['pct_dataset_labeled'].apply(to_float2) + r'\%'
    d2['Examples Processed'] = (
        d['num_img_patches_processed'].apply(to_int)
        + r' (' +
        ((d['num_img_patches_processed'] - baseline_num_processed) *100 / baseline_num_processed).apply(lambda x: '%+0.2f' % x)
        + r'\%)'
    )
    def bold_if_startswith(expected_prefix):
        def _bold(strng):
            if strng.startswith(expected_prefix):
                return r'\textbf{%s}' % strng
            else:
                return strng
        return _bold

    to_latex_formatters = {
        'Examples Processed': bold_if_startswith(to_int(
            d['num_img_patches_processed'].min())),
        'Test Accuracy': bold_if_startswith(to_float2(
            d['val_acc'].max()*100)),
        'Percent Labeled': bold_if_startswith(to_float2(
            d['pct_dataset_labeled'].min()))
    }

    fout.write(d2.reset_index().to_latex(formatters=to_latex_formatters, escape=False, index=False))
    print(d2.reset_index().to_latex(formatters=to_latex_formatters, escape=False, index=False))
    print(d.to_string())

[ax.plot(xy[1], xy[0], marker, markersize=ms,
         color=color if color2 is None else color2, alpha=1)
 for xy, color in keypoints
 for marker, ms, color2 in [('P', 25, None), ('+', 35, 'white')]]
#  [ax.add_patch(patches.Ellipse([xy[0], xy[1]], 0.0075, 6, facecolor='none',
                              #  linewidth=15, alpha=.8, edgecolor=color, lw=4))
 #  for xy, color in keypoints]

f.savefig(join(analysis_dir, 'topn_best_val_accs_per_experiment.png'))

# useless bar plot
#  _bpm1 = dfo[['val_acc']]\
#      .unstack('Experiment').max().rename('val_acc').to_frame().T.droplevel(0, axis=1)\
#      .join(dfm20[['val_acc']].max().rename('MedAL (patience=20)'))\
#      .join(dfb[['val_acc']].max().rename('ResNet18'))\
#      .T
#  _bpm2 = dfo[['pct_dataset_labeled']]\
#      .unstack('Experiment').max().rename('pct_dataset_labeled').to_frame().T.droplevel(0, axis=1)\
#      .join(dfm20[['pct_dataset_labeled']].max().rename('MedAL (patience=20)'))\
#      .join(pd.Series({'pct_dataset_labeled': 100.0}, name='ResNet18'))\
#      .T
#  bpm = pd.concat([_bpm1, _bpm2], axis=1)

#  f, (ax1, ax2) = plt.subplots(2, 1)
#  bpm.drop('pct_dataset_labeled', axis=1).plot.bar(legend=False, ax=ax1, rot=30, ylim=(.6, 1))
#  ax2.table(
#      cellText=bpm.round(4).sort_values('val_acc', ascending=False)\
#      .reset_index().values,
#      colLabels=['Model', 'Best Test Acc', 'Percent dataset labeled'],
#      loc='center')
#  ax2.axis('tight')
#  ax2.axis('off')
#  f.tight_layout()
#  f.subplots_adjust(top=0.92, hspace=.55)
#  f.suptitle("Best Test Accuracy")
#  f.savefig(join(analysis_dir, 'best_model_val_acc.png'))


dfbb = {
    "ResNet18": "data/_analysis/R6-20190315T030959.log/logdata.csv",
    "InceptionV3": "data/_analysis/A6e-20190222T103123.log/logdata.csv",
}
dfbb = pd.concat({k: pd.read_csv(v) for k, v in dfbb.items()}, sort=False)\
    .query('perf and al_iter == 0 and epoch <= 150')\
    .droplevel(1).set_index('epoch', append=True).unstack(level=0)

ax = dfbb['val_acc'].plot()
ax.legend(loc='center right')
table = pd.plotting.table(
    ax, dfbb['val_acc'].describe().round(4).loc[['max', 'min']],
    loc='lower center', colWidths=[0.3, 0.3, 0.3], alpha=1)
table.auto_set_font_size(False)

ax.set_xlabel('Epoch')
ax.set_ylabel('Test Accuracy')
f = ax.figure
f.suptitle("Test Accuracy vs Epoch")
#  f.tight_layout(rect=[0, 0.03, 1, 0.95])
f.savefig(join(analysis_dir, "baselines_acc_vs_epoch.png"))

import IPython ; IPython.embed() ; import sys ; sys.exit()
