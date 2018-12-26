import argparse
import os

import numpy as np

PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
EMBS_ROOT = os.path.join(PROJ_ROOT, 'data', 'embs')
EXPER_ROOT = os.path.join(PROJ_ROOT, 'experiments')

if __name__ != '__main__':
    exit()

#======================================================================================
parser = argparse.ArgumentParser()
parser.add_argument('models', nargs='+')
parser.add_argument('--emb-dim', default=64, type=int)
parser.add_argument('-o', '--out', default='ensemble')
args = parser.parse_args()
#======================================================================================

out_dir = os.path.join(EMBS_ROOT, args.out)
os.mkdir(os.path.join(EMBS_ROOT, args.out))

for dtype in ['img', 'instr']:

    model_embs = []
    for model_epoch in args.models:
        model, epoch = model_epoch.split(',')
        emb_file = f'{dtype}_embs_{epoch}.npy'
        embs = np.load(os.path.join(EXPER_ROOT, model, 'run', emb_file))
        model_embs.append(embs)

    avg_embs = np.mean(model_embs, axis=0)

    np.save(os.path.join(out_dir, f'{dtype}_embs_avg.npy'), avg_embs)
