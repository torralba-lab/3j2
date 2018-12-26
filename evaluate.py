if __name__ != '__main__':
    exit()

import argparse
import os
import pickle

from tqdm import tqdm

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as nnf
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

import dataset
import model

#===============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('epoch', type=int)
parser.add_argument('--batch-size', default=64, type=int)
args = parser.parse_args()
#===============================================================================

with open('run/opts.pkl', 'rb') as f_opts:
    orig_args = pickle.load(f_opts)
    for k, v in vars(args).items():
        setattr(orig_args, k, v)
args = orig_args

n_gpu = torch.cuda.device_count()
setattr(args, 'batch_size', int(args.batch_size / n_gpu) * n_gpu)

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

varargs = vars(args)

#===============================================================================

ds_test = dataset.create(part='test', **varargs)

loader_opts = {'batch_size': args.batch_size, 'shuffle': False,
               'pin_memory': True, 'num_workers': args.nworkers}
test_loader = torch.utils.data.DataLoader(ds_test, **loader_opts)

net = model.create(**varargs)
net.load_state_dict(torch.load(f'run/snaps/model_{args.snap}.pth'))

inputs = {k: Variable(inp.cuda()) for k, inp in net.create_inputs().items()}

if n_gpu > 1:
    net = nn.DataParallel(net)
net = net.cuda()

net.eval()
img_embs = np.empty((len(ds_test), args.emb_dim), dtype='float32')
instr_embs = np.empty((len(ds_test), args.emb_dim), dtype='float32')
n = 0
for batch_idx, cpu_inputs in enumerate(tqdm(test_loader), 1):
    for k, v in inputs.items():
        ct = cpu_inputs[k]
        v.data.resize_(ct.size()).copy_(ct)
        v.volatile = True

    outputs = net(**inputs)
    batch_img_embs, batch_instr_embs = outputs[-2:]
    bsl = slice(n, n+len(batch_img_embs))
    img_embs[bsl] = batch_img_embs.data.cpu().numpy()
    instr_embs[bsl] = batch_instr_embs.data.cpu().numpy()
    n += len(batch_img_embs)

np.save('run/instr_embs.npy', instr_embs)
np.save('run/img_embs.npy', img_embs)
