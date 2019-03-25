import pandas as pd
import seaborn as sns
import re
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.ticker as mticker
from os.path import join


# define log csv data and dataset parameters not available in csvs
analysis_dir = './data/analysis'  # dir to save plots to
points_to_label_per_al_iter = 20
train_set_size = 949

fp_baseline = "data/_analysis/R6-20190315T030959.log/logdata.csv"
fp_medal = "data/_analysis/RM6-20190319T005512.log/logdata.csv"
fps_varying_online_frac = [
    "data/_analysis/RMO6-0d-20190323T003224.log/logdata.csv",
    "data/_analysis/RMO6-100d-20190323T082726.log/logdata.csv",
    "data/_analysis/RMO6-12.5d-20190322T142710.log/logdata.csv",
    "data/_analysis/RMO6-25d-20190323T152428.log/logdata.csv",
    "data/_analysis/RMO6-37.5d-20190322T051602.log/logdata.csv",
    "data/_analysis/RMO6-50d-20190323T030509.log/logdata.csv",
    "data/_analysis/RMO6-62.5d-20190322T092309.log/logdata.csv",
    "data/_analysis/RMO6-75d-20190323T185418.log/logdata.csv",
    "data/_analysis/RMO6-87.5d-20190322T173655.log/logdata.csv",
]


def get_train_frac(fp):
    return re.sub(r'.*RMO6-([\d\.]+).?-.*', r'\1', fp)


# load the data
dfb = pd.read_csv(fp_baseline).query('perf').set_index('epoch')
dfm = pd.read_csv(fp_medal).query('perf')
dfs = {'Online - ' + get_train_frac(fp): pd.read_csv(fp)
       for fp in fps_varying_online_frac}
dfo = pd.concat(dfs, names=['Experiment', 'log_line_num'])
dfo['online_sample_frac'] = dfo.index.get_level_values('Experiment')\
    .str.extract('Online - (.*)').astype('float').values
# --> reindex dfm and dfo to guarantee if the run was incomplete we show empty
# space in plot that would represent usage of the full dataset.
_mi = pd.MultiIndex.from_product([
        np.arange(1, 49), np.arange(1, 151)], names=['al_iter', 'epoch'])
dfm = dfm.set_index(['al_iter', 'epoch']).reindex(_mi).reset_index()

#  import IPython ; IPython.embed() ; import sys ; sys.exit()
#  _mi = pd.concat({fp: pd.Series(index=_mi)
                 #  for fp in dfo.index.levels[0]})
#  _mi.index.set_names('fp', level=0, inplace=True)
#  _mi = _mi.index
#  dfo.reset_index().set_index(['fp', 'al_iter', 'epoch']).reindex(_mi).reset_index().set_index(['fp', 'log_line_num'])


# compute percent data seen (for x axis of plot)
for df in [dfo, dfm]:
    N = df['al_iter'].values * points_to_label_per_al_iter
    df['pct_dataset_labeled_int'] = (N / train_set_size * 100).astype(int)
    x = (N / train_set_size * 100)
    x[x > 100] = 100  # clip ends, since the reindexing operation would make seem like over 100%
    df['pct_dataset_labeled'] = x

dfb['num_img_patches_processed'] = dfb.index * train_set_size
dfm['num_img_patches_processed'] = \
    (dfm['al_iter'].values * points_to_label_per_al_iter).cumsum()
dfo['num_img_patches_processed'] = \
    (points_to_label_per_al_iter
     + dfo['online_sample_frac']/100
     * dfo['al_iter'] * points_to_label_per_al_iter)\
    .unstack('Experiment').cumsum().stack('Experiment')\
    .swaplevel().sort_index()

# plot 1: val acc vs percent dataset labeled
# --> prepare results for plot
def main_perf_plot(subset_experiments=()):
    medalpltdata = dfm\
        .set_index(['pct_dataset_labeled_int', 'epoch'])['val_acc']\
        .rename('MedAL')
    onlinepltdata = dfo\
        .query('perf')\
        .set_index(['pct_dataset_labeled_int', 'epoch'], append=True)\
        .drop('al_iter', axis=1)\
        .droplevel('log_line_num')['val_acc']\
        .unstack('Experiment')\
        .reindex(medalpltdata.index)
    if subset_experiments:
        onlinepltdata = onlinepltdata[subset_experiments]
    # --> add the plots
    axs = onlinepltdata\
        .plot(ylim=(0, 1), color='red', subplots=True, figsize=(10, 8))
    [medalpltdata.plot(ax=ax, alpha=.5, color='green', linewidth=0.3,
                    label='_nolegend_') for ax in axs]
    [ax.legend(loc='lower right', frameon=False) for ax in axs]
    [ax.hlines(dfb['val_acc'].max(), 0, medalpltdata.shape[0],
            color='lightblue', linestyle='--') for ax in axs]
    # --> handle xticks.
    axs[-1].set_xlabel('Percent Dataset Labeled')
    axs[len(axs)//2].set_ylabel('Validation Accuracy')
    axs[0].xaxis.set_major_locator(mticker.LinearLocator(10))
    axs[0].xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, pos: medalpltdata.index[int(x)][0]))
    axs[0].figure.savefig(join(analysis_dir, 'varying_online_frac%s.png'
                               % len(subset_experiments)))
main_perf_plot()
main_perf_plot([
    'Online - 0', 'Online - 37.5', 'Online - 87.5', 'Online - 100'])
main_perf_plot(['Online - 87.5'])


# get one row per experiment per al iter.
Z = dfo\
    .groupby(['online_sample_frac', 'al_iter'])\
    .agg({'pct_dataset_labeled': 'first',
          'num_img_patches_processed': 'max',
          'val_acc': 'max'})\
    .reset_index()


# plot 2: training time (number of image patches used)
# --> plot amount of data used as we change the sample frac
f, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=False, figsize=(10, 8))
tmp = Z.set_index(['pct_dataset_labeled', 'online_sample_frac'])\
    ['num_img_patches_processed']\
    .unstack('online_sample_frac')\
    .join(dfm.groupby('pct_dataset_labeled')['num_img_patches_processed'].max()
          .rename('MedAL'), how='outer')
tmp.plot(ax=ax1)
tmp.drop('MedAL', axis=1).plot(ax=ax2)
for ax in [ax1, ax2]:
    ax.hlines(dfb['num_img_patches_processed'].max(),
            0, 100, color='lightblue', linestyle='--', label='Baseline ResNet')
    ax.legend()
ax1.set_ylabel('Number of Training Examples Processed')
ax1.set_xlabel('Percent Dataset Labeled')
ax2.set_xlabel('Percent Dataset Labeled')
f.savefig(join(analysis_dir, 'num_img_patches_processed.png'))
# --> plot showing amount of data that vanilla medal uses to train (for slides)
#  ax = dfm.groupby('pct_dataset_labeled')['num_img_patches_processed'].max().rename('MedAL').plot()
#  ax.hlines(dfb['num_img_patches_processed'].max(),
#          0, 100, color='lightblue', linestyle='--', label='Baseline ResNet')
#  ax.legend()
#  ax.set_ylabel('Image patches processed')
#  ax.set_xlabel('Percent Dataset Labeled')
#  plt.show()

# plot 3: best performing model
g = dfo.groupby('Experiment')\
    .apply(lambda x: x.sort_values(['val_acc', 'pct_dataset_labeled'],
                                   ascending=False).head(15))\
    .sort_values('online_sample_frac')\
    [['val_acc', 'pct_dataset_labeled', 'al_iter', 'online_sample_frac']]\
    .droplevel(0).droplevel('log_line_num').reset_index()
f, ax = plt.subplots(1, 1, figsize=(10, 8))
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
    'val_acc', 'pct_dataset_labeled', hue='Experiment', data=g, ax=ax,
    #  palette=sns.palplot(sns.color_palette("cubehelix", 9))
    #  palette=sns.palplot(sns.color_palette("coolwarm", 9))
    palette=sns.palplot(sns.color_palette("hsv", 9))
    #  palette='GnBu_d')
)
ax.vlines(dfb['val_acc'].max(), 0, 100, linestyle='--',
          color='lightblue', alpha=.8, label='ResNet18 (best accuracy)')
ax.vlines(dfm['val_acc'].max(), 0, 100, linestyle='--', alpha=.5,
          label='MedAL (best accuracy)')
ax.plot(*dfm.loc[dfm['val_acc'].idxmax()]\
         .loc[['val_acc', 'pct_dataset_labeled']].values,
         marker='x', color='green')
ax.legend(framealpha=.8, loc='lower center')


ax.set_title("Top 15 highest validation accuracies for each experiment")
f.savefig(join(analysis_dir, 'topn_best_val_accs_per_experiment.png'))

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
table = pd.plotting.table(
    ax, dfbb['val_acc'].describe().round(4).loc[['max', 'min']],
    loc='lower center', colWidths=[0.2, 0.2, 0.2], alpha=1)
table.auto_set_font_size(False)
f = ax.figure
f.suptitle("Validation Accuracy vs Epoch")
#  f.tight_layout(rect=[0, 0.03, 1, 0.95])
f.savefig(join(analysis_dir, "baselines_acc_vs_epoch.png"))

import IPython ; IPython.embed() ; import sys ; sys.exit()
