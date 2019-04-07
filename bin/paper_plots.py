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
fp_medal = "data/_analysis/RM6e-20190324T140934.log/logdata.csv"
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
    return re.sub(r'.*RMO6-([\d\.]+).?-.*', r'\1', fp)


# load the data
dfb = pd.read_csv(fp_baseline).query('perf').sort_values('epoch').set_index('epoch')
dfm = pd.read_csv(fp_medal).query('perf').sort_values(['al_iter', 'epoch'])
dfs = {'Online - ' + get_train_frac(fp): pd.read_csv(fp).query('perf')
       for fp in fps_varying_online_frac}
dfo = pd.concat(dfs, names=['Experiment', 'log_line_num'])
dfo['online_sample_frac'] = dfo.index.get_level_values('Experiment')\
    .str.extract('Online - (.*)').astype('float').values
dfo = dfo.sort_values(['online_sample_frac', 'al_iter', 'epoch'])
# --> reindex dfm and dfo to guarantee if the run was incomplete we show empty
# space in plot that would represent usage of the full dataset.
_mi = pd.MultiIndex.from_product([
        np.arange(1, 49), np.arange(1, 151)], names=['al_iter', 'epoch'])
#  dfm = dfm.set_index(['al_iter', 'epoch']).reindex(_mi).reset_index()
assert dfm.dropna(axis=1, how='all').dropna().groupby('al_iter').count()['epoch'].min() >= dfo['epoch'].max()

# compute percent data labeled (for x axis of plot)
for df in [dfo, dfm]:
    N = df['al_iter'].values * points_to_label_per_al_iter
    df['pct_dataset_labeled_int'] = (N / train_set_size * 100).astype(int)
    x = (N / train_set_size * 100)
    #  x[x > 100] = 100  # clip ends, since the points_to_label_per_al_iter is not divisble by train_set_size and since the reindexing operation would make seem like over 100%
    df['pct_dataset_labeled'] = x

# compute num examples processed.  medal and omedal have +1 because initial
# train set is 1 + points_to_label_per_al_iter.
dfb['num_img_patches_processed'] = dfb.index * train_set_size
dfm['num_img_patches_processed'] = \
    (dfm['al_iter'] * points_to_label_per_al_iter + 1).cumsum()
dfo['num_img_patches_processed'] = \
    (points_to_label_per_al_iter
     + np.floor(
         dfo['online_sample_frac']/100
         * (1 + (dfo['al_iter']-1) * points_to_label_per_al_iter)))\
    .unstack('Experiment').cumsum().stack('Experiment')\
    .swaplevel().sort_index()

# plot 1: val acc vs percent dataset labeled
# --> prepare results for plot
def main_perf_plot(subset_experiments=(), add_medal_to_legend=False):
    medalpltdata = dfm\
        .set_index(['pct_dataset_labeled_int', 'epoch'])['val_acc']\
        .rename('MedAL')
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
    axs = onlinepltdata\
        .plot(ylim=(0, 1), color='red', subplots=True, figsize=(8, 8))
    if add_medal_to_legend:
        [medalpltdata.plot(ax=ax, alpha=.6, color='green', linewidth=0.4)
         for ax in axs]
    else:
        [medalpltdata.plot(ax=ax, alpha=.6, color='green', linewidth=0.4,
                           label='_nolegend_')
         for ax in axs]
    [ax.legend(loc='lower left', frameon=True) for ax in axs]
    [ax.hlines(dfb['val_acc'].max(), 0, medalpltdata.shape[0],
            color='dodgerblue', linestyle='--') for ax in axs]
    # --> handle xticks.
    axs[-1].set_xlabel('Percent Dataset Labeled')
    axs[len(axs)//2].set_ylabel('Validation Accuracy')
    axs[0].xaxis.set_major_locator(mticker.LinearLocator(10))
    axs[0].xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, pos: min(100, medalpltdata.index[int(x)][0])))
    axs[0].figure.savefig(join(analysis_dir, 'varying_online_frac%s.png'
                               % len(subset_experiments)))
main_perf_plot()
main_perf_plot([
    'Online - 0', 'Online - 37.5', 'Online - 87.5', 'Online - 100'])
main_perf_plot(['Online - 87.5'], add_medal_to_legend=True)


# get one row per experiment per al iter.
Z = dfo\
    .groupby(['online_sample_frac', 'al_iter'])\
    .agg({'pct_dataset_labeled': 'first',
          'num_img_patches_processed': 'max',
          'val_acc': 'max'})\
    .reset_index()
baseline_num_processed = dfb['num_img_patches_processed'].max()
Z['process_more_pts_than_baseline'] = Z['num_img_patches_processed'] > baseline_num_processed
Z['val_acc_worse_than_baseline'] = Z['val_acc'] < dfb['val_acc'].max()


# get keypoints for next couple plots
_tmp = dfo[['val_acc', 'pct_dataset_labeled', 'num_img_patches_processed']]
keypoints = [
    # online experiments
    (_tmp.loc[dfo['val_acc'].idxmax()], 'dimgray'),
    (_tmp.loc[dfo.query('online_sample_frac == 37.5')['val_acc'].idxmax()],
     'red'),
    (_tmp.loc[dfo.query('online_sample_frac == 87.5')
             .sort_values('val_acc', ascending=False)
             .head(5)['pct_dataset_labeled'].idxmin()],
     'dimgray'),
    (_tmp.loc[ _tmp.loc[_tmp['val_acc'] >= dfb['val_acc'].max()]['num_img_patches_processed'].idxmin()],
     'red'),
    # medal
    (dfm.loc[dfm['val_acc'].idxmax()].loc[['val_acc', 'pct_dataset_labeled', 'num_img_patches_processed']]\
     .rename(('MedAL', '')), 'black')
]

# plot 2: training time (number of image patches used)
# --> plot amount of data used as we change the sample frac
def plot_training_time(logy=True, fracs=None, use_keypoints=True):
    f, ax = plt.subplots(1, 1, figsize=(6, 4))
    tmp = Z.set_index(['pct_dataset_labeled', 'online_sample_frac'])\
        ['num_img_patches_processed']\
        .unstack('online_sample_frac')\
        .join(dfm.groupby('pct_dataset_labeled')['num_img_patches_processed']
              .max().rename('MedAL'), how='outer')
    # plot exponential curves for experiments and MedAL
    tmp.drop('MedAL', axis=1).plot(ax=ax, logy=logy, legend=False)
    tmp['MedAL'].plot(ax=ax, style='-.', logy=logy, color='black', legend=False)

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
        points = points.append(Z.loc[ Z.loc[Z['val_acc'] >= dfb['val_acc'].max()]['num_img_patches_processed'].idxmin()])

        #  add medal to points
        points = points.append(dfm.loc[dfm['val_acc'].idxmax()])

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
        [ax.plot(xy[1], xy[2], marker, markersize=ms, color=color)
        for xy, color in keypoints for marker, ms in [('+', 35), ('P', 15)]]

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
plot_training_time(logy=True, fracs=['87.5', '37.5'], use_keypoints=False)

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
topn = 5
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
    'Percent Dataset Labeled', 'Validation Accuracy', hue='Experiment',
    data=g.rename({
        'pct_dataset_labeled': 'Percent Dataset Labeled',
        'val_acc': 'Validation Accuracy', }, axis=1), ax=ax,
    #  palette=sns.palplot(sns.color_palette("cubehelix", 9))
    #  palette=sns.palplot(sns.color_palette("coolwarm", 9))
    palette=sns.palplot(sns.color_palette("hsv", 9))
    #  palette='GnBu_d')
)
ax.hlines(dfb['val_acc'].max(), 0, 100, linestyle='--',
          color='dodgerblue', alpha=.8, label='ResNet18 (best accuracy)')
ax.hlines(dfm['val_acc'].max(), 0, 100, linestyle='--', alpha=.5,
          color='black', label='MedAL (best accuracy)')
ax.legend(framealpha=.7, loc='left center', ncol=1)
ax.set_title("Top %s Highest Validation Accuracies For Each Experiment" % topn)
# add annotations to key points of interest on plot
with open(join(analysis_dir, 'topn_keypoint_table.tex'), 'w') as fout:
    print("\n\n  ---  keypoint table  ---")
    d = pd.DataFrame([x[0] for x in keypoints])
    d.index = pd.MultiIndex.from_tuples(d.index)
    d = d.reset_index(level=1, drop=True)
    d.index.name = 'Experiment'
    fout.write(d.to_latex())
    print(d.to_string())

[ax.plot(xy[1], xy[0], marker, markersize=ms, color=color)
for xy, color in keypoints for marker, ms in [('+', 35), ('P', 15)]]
#  [ax.add_patch(patches.Ellipse([xy[0], xy[1]], 0.0075, 6, facecolor='none',
                              #  linewidth=15, alpha=.8, edgecolor=color, lw=4))
 #  for xy, color in keypoints]

f.savefig(join(analysis_dir, 'topn_best_val_accs_per_experiment.png'))

# useless bar plot
_bpm1 = dfo[['val_acc']]\
    .unstack('Experiment').max().rename('val_acc').to_frame().T.droplevel(0, axis=1)\
    .join(dfm[['val_acc']].max().rename('MedAL'))\
    .join(dfb[['val_acc']].max().rename('ResNet18'))\
    .T
_bpm2 = dfo[['pct_dataset_labeled']]\
    .unstack('Experiment').max().rename('pct_dataset_labeled').to_frame().T.droplevel(0, axis=1)\
    .join(dfm[['pct_dataset_labeled']].max().rename('MedAL'))\
    .join(pd.Series({'pct_dataset_labeled': 100.0}, name='ResNet18'))\
    .T
bpm = pd.concat([_bpm1, _bpm2], axis=1)

f, (ax1, ax2) = plt.subplots(2, 1)
bpm.drop('pct_dataset_labeled', axis=1).plot.bar(legend=False, ax=ax1, rot=30, ylim=(.6, 1))
ax2.table(
    cellText=bpm.round(4).sort_values('val_acc', ascending=False)\
    .reset_index().values,
    colLabels=['Model', 'Best Validation Acc', 'Percent dataset labeled'],
    loc='center')
ax2.axis('tight')
ax2.axis('off')
f.tight_layout()
f.subplots_adjust(top=0.92, hspace=.55)
f.suptitle("Best Validation Accuracy")
f.savefig(join(analysis_dir, 'best_model_val_acc.png'))


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
ax.set_ylabel('Validation Accuracy')
f = ax.figure
f.suptitle("Validation Accuracy vs Epoch")
#  f.tight_layout(rect=[0, 0.03, 1, 0.95])
f.savefig(join(analysis_dir, "baselines_acc_vs_epoch.png"))

import IPython ; IPython.embed() ; import sys ; sys.exit()

# TODO:
# the model that achieves same or better accuracy than baseline, but uses min
# computation
Z.loc[ Z.loc[Z['val_acc'] >= dfb['val_acc'].max()]['num_img_patches_processed'].idxmin()]
#  (baseline_num_processed - Z.loc[Z['val_acc'] >= dfb['val_acc'].max()]['num_img_patches_processed'].min() ) / baseline_num_processed
