import argparse
import os
import pickle
import sys

from tqdm import tqdm

import numpy as np

import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable

import dataset

PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_ROOT = os.path.join(os.path.dirname(__file__), 'data')
EMBS_ROOT = os.path.join(os.path.dirname(__file__), 'embs')
EXP_ROOT = os.path.join(PROJ_ROOT, 'experiments')
sys.path.append(PROJ_ROOT)


if __name__ != '__main__':
    exit()

#===============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('experiment')
parser.add_argument('epoch', type=int)
parser.add_argument('--batch-size', default=128, type=int)
args = parser.parse_args()
#===============================================================================
exp_dir = os.path.join(EXP_ROOT, args.experiment)
run_dir = os.path.join(exp_dir, 'run')
sys.path.insert(0, exp_dir)
import model

with open(os.path.join(run_dir, 'opts.pkl'), 'rb') as f_opts:
    snap_args = pickle.load(f_opts)

n_gpu = torch.cuda.device_count()
setattr(snap_args, 'batch_size', int(args.batch_size / n_gpu) * n_gpu)

snap_varargs = vars(snap_args)
#===============================================================================

embs_dir = os.path.join(EMBS_ROOT, f'{args.experiment}_{args.epoch}')
os.makedirs(embs_dir, exist_ok=True)

net = model.create(**snap_varargs)
net.load_state_dict(torch.load(os.path.join(run_dir, f'snaps/model_{args.epoch}.pth')))

inputs = {k: Variable(inp.cuda(), volatile=True) for k, inp in net.create_inputs().items()}

net = net.eval().cuda()

embers = {
    'imgs': net.img_emb,
    'instrs': net.instr_emb,
}

loader_opts = {'batch_size': args.batch_size, 'shuffle': False,
               'pin_memory': True, 'num_workers': snap_args.nworkers}
snap_varargs['dataset'] = os.path.join(DATA_ROOT, 'recipes')
for part in ['test', 'val', 'train']:
    ds = dataset.create(part=part, **snap_varargs)
    for dtype in ['instrs', 'imgs']:
        ds.datatype = dtype
        out_path = os.path.join(embs_dir, f'{dtype[:-1]}_embs_{part}.dat')
        embs = np.memmap(out_path, dtype='float32', mode='w+',
                             shape=(len(ds), snap_args.emb_dim))
        loader = torch.utils.data.DataLoader(ds, **loader_opts)

        ember = embers[dtype]

        n = 0
        inp = inputs[dtype]
        for batch_idx, cpu_inputs in enumerate(tqdm(loader, desc=f'{part} {dtype}'), 1):
            ct = cpu_inputs[dtype]
            inp.data.resize_(ct.size()).copy_(ct)

            batch_embs = ember(inp)
            bsl = slice(n, n+len(batch_embs))
            embs[bsl] = batch_embs.data.cpu().numpy()
            n += len(batch_embs)

        del embs
