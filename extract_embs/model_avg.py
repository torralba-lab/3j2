import argparse
import os

import numpy as np

EMBS_ROOT = os.path.join(os.path.dirname(__file__), 'embs')

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

for part in ['test', 'val', 'train']:
    for dtype in ['img', 'instr']:
        emb_file = f'{dtype}_embs_{part}.dat'

        model_embs = []
        for model in args.models:
            model_dir = os.path.join(EMBS_ROOT, model)
            embs = np.memmap(os.path.join(model_dir, emb_file),
                             dtype='float32', mode='r')
            embs = embs.reshape(-1, args.emb_dim)
            model_embs.append(embs)

        avg_embs = np.memmap(os.path.join(out_dir, emb_file), mode='w+',
                             dtype='float32', shape=model_embs[0].shape)
        avg_embs[:] = np.mean(model_embs, axis=0)
        del avg_embs
