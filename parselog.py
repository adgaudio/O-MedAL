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

fp_in = sys.argv[1]
img_dir = sys.argv[2]
assert exists(img_dir)

print("Analyzing log:  %s" % fp_in)
print("Saving images to directory:  %s" % img_dir)

regex_epoch = r'Epoch (\d+)/\d+$'
regex_pat1 = (
    r'^ *(?P<iteration>\d+)/\d+ \[[=\.>]+\] - .*'
    r'loss: (?P<train_loss>\d+\.\d+(e-?\d+)?) - '
    r'acc: (?P<train_acc>\d+.\d+(e-?\d+)?)')
regex_pat2 = (
    r' - val_loss: (?P<val_loss>\d+.\d+(e-?\d+)?)'
    r' - val_acc: (?P<val_acc>\d+.\d+(e-?\d+)?)')
perf = []
with open(fp_in, 'r') as fin:
    epoch = None
    for line in fin:
        m0 = re.match(regex_epoch, line)
        if m0:
            epoch = m0.group(1)
        m = re.search(regex_pat1, line)
        if m:
            dct = {'epoch': epoch}
            dct.update(m.groupdict())
            m2 = re.search(regex_pat2, line)
            if m2:
                dct.update(m2.groupdict())
            else:
                dct.update({"val_loss": np.nan, "val_acc": np.nan})
            perf.append(
                {k: float(v) if k not in {"epoch", "iteration"} else int(v)
                 for k, v in dct.items()})

df = pd.DataFrame(perf)
assert not df.empty, "Error parsing the log file.  Found no usable data."
df['perf'] = ~df['val_acc'].isnull()

# sanity check data:
assert (df['val_acc'].isnull() == df['val_loss'].isnull()).all()
if not (df['epoch'].value_counts() == df['epoch'].value_counts().max()).all():
    print(
        "\n\nWARNING: \n Run may not have completed successfully."
        " At least one epoch does not have enough iterations\n")


#  Run some plots
df.query('perf')[['train_loss', 'val_loss']].plot(title="Loss vs Epoch")\
    .figure.savefig(join(img_dir, "loss_vs_epoch.png"))
df.query('perf')[['train_acc', 'val_acc']].plot(title="Accuracy vs Epoch")\
    .figure.savefig(join(img_dir, "accuracy_vs_epoch.png"))

f, (ax1, ax2) = plt.subplots(2, 1)
f.suptitle("Epoch vs Iteration: Train Performance")
ax1.set_title("Training Accuracy")
sns.heatmap(df.pivot('iteration', 'epoch', 'train_acc'), ax=ax1)
ax2.set_title("Training Loss")
sns.heatmap(df.pivot('iteration', 'epoch', 'train_loss'), ax=ax2)
f.tight_layout(rect=[0, 0.03, 1, 0.95])
f.savefig(join(img_dir, "iteration_vs_epoch.png"))
