import argparse
import os
import re
import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import median_filter

PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_ROOT = os.path.join(PROJ_ROOT, 'data')
EXP_ROOT = os.path.join(PROJ_ROOT, 'experiments')
sys.path.append(PROJ_ROOT)

def read_stats(exp_name, stats_re, val=False):
    log_path = os.path.join(EXP_ROOT, exp_name, 'run', 'log.txt')
    ts = []
    losses = []
    epoch_ts = []
    with open(log_path) as f_stats:
        for l in f_stats:
            m = stats_re.match(l.rstrip())
            if not m:
                continue
            if val:
                epoch, loss = m.groups()
                t = int(epoch)
            else:
                epoch, itr, itr_per_epoch, loss = m.groups()
                t = (int(epoch) - 1)*int(itr_per_epoch) + int(itr)
                if itr == itr_per_epoch:
                    epoch_ts.append(t)
            ts.append(t)
            losses.append(float(loss))

    return ts, losses, epoch_ts

if __name__ != '__main__':
    exit()

#======================================================================================
parser = argparse.ArgumentParser()
parser.add_argument('exp_names', nargs='+')
parser.add_argument('--stat', choices=('emb', 'topic', 'pdist'), default='emb')
parser.add_argument('--val', action='store_true')
parser.add_argument('--legend-names', nargs='+', default=[])
parser.add_argument('--xlim')
parser.add_argument('--savefig')
args = parser.parse_args()
#======================================================================================

if args.val:
    stats_re = re.compile(r'\[([1-9][0-9]*)\] \(VAL\).*\|.*loss: .*%s=(\d+\.\d+)' % args.stat)
else:
    stats_re = re.compile(r'\[([1-9][0-9]*)\] \((\d+)/(\d+)\).*\|.*loss: .*%s=(\d+\.\d+)' % args.stat)

plt.figure()

min_loss = float('inf')
max_loss = 0
epoch_ts = []
exp_max_iter = {}
if not args.legend_names:
    args.legend_names = args.exp_names
for exp_name, legend_name in zip(args.exp_names, args.legend_names):
    ts, losses, ets = read_stats(exp_name, stats_re, val=args.val)
    exp_max_iter[exp_name] = max(ts)
    if len(ets) > len(epoch_ts):
        epoch_ts = ets
    min_loss = min(min_loss, *losses)
    max_loss = max(max_loss, *losses)
    if not args.val:
        losses = median_filter(losses, size=20, mode='mirror')
    plt.plot(ts, losses, label=legend_name)

plt.vlines(epoch_ts, ymin=min_loss, ymax=max_loss,
           linestyles='dashed', linewidth=1)

plt.xlabel('iter')
plt.ylabel('loss')
plt.legend()

if args.xlim == 'min':
    plt.xlim(None, min(exp_max_iter.values()))
elif args.xlim:
    plt.xlim(None, exp_max_iter[args.xlim])

if args.savefig is not None:
    plt.savefig(f'{args.savefig}.eps', bbox_inches='tight')
plt.show()
plt.close()
