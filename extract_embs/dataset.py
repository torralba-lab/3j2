import os
import pickle
import random

from PIL import Image
import numpy as np
import torch
import torch.utils.data
import torchvision.transforms as transforms


PAD, UNK, EOS = 0, 1, 2
EOS_TOKS = {'.', '!'}

class RecipeDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, vocab, img_root, vocab_size, max_seqlen,
                 part='train', **kwargs):
        super(RecipeDataset, self).__init__()

        self.img_root = img_root
        self.max_seqlen = max_seqlen
        self._datatype = 'instrs'

        txforms = [
            transforms.Scale(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
        self.transform = transforms.Compose(txforms)

        with open(f'{vocab}', 'rb') as f_vocab:
            vocab = pickle.load(f_vocab)
        self.vocab = ['PAD', 'UNK', '</s>'] + vocab[:vocab_size-3]
        self.w2i = {w: i for i, w in enumerate(self.vocab)}

        with open(f'{dataset}_{part}.pkl', 'rb') as f_ds:
            self.recipes = pickle.load(f_ds)

        self.imgs = []
        for i, recipe in enumerate(self.recipes):
            for j, img in enumerate(recipe.get('images', [])):
                self.imgs.append((i, j))

    @property
    def datatype(self): return self._datatype

    @datatype.setter
    def datatype(self, datatype):
        assert datatype in {'instrs', 'imgs'}
        self._datatype = datatype

    def __getitem__(self, index):
        if self.datatype == 'imgs':
            r_idx, im_idx = self.imgs[index]
            recipe = self.recipes[r_idx]
            imid = recipe['images'][im_idx]
            datum = {
                'ids': imid,
                'imgs': self.transform(Image.open(os.path.join(self.img_root, imid))),
            }
        elif self.datatype == 'instrs':
            recipe = self.recipes[index]
            instr_toks = torch.LongTensor(self.max_seqlen).zero_()
            toks = []
            for instr in recipe['instructions']:
                for tok in instr:
                    if len(toks) == self.max_seqlen:
                        break
                    toks.append(self.w2i.get(tok, UNK))
                if len(toks) == self.max_seqlen:
                    break
                if tok not in EOS_TOKS:
                    toks.append(EOS)
            tok_offset = max(0, int((self.max_seqlen - len(toks)) / 2) - 1)
            for i, tok in enumerate(toks, tok_offset):
                instr_toks[i] = tok
            datum = {
                'ids': recipe['id'],
                'instrs': instr_toks,
            }

        return datum

    def __len__(self):
        return len(self.imgs) if self.datatype == 'imgs' else len(self.recipes)

def create(*args, **kwargs):
    return RecipeDataset(*args, **kwargs)

if __name__ == '__main__':
    ds_opts = {
        'dataset': 'data/recipes',
        'doctops': '../data/lda_20_doctops',
        'vocab': '../data/vocab.pkl',
        'img_root': '../data/images',
        'max_seqlen': 150,
        'vocab_size': 15000,
        'datatype': 'instrs',
    }
    ds_test = RecipeDataset(**ds_opts, part='test')
    ds_test.datatype = 'imgs'
    # print(len(ds_test))
    datum = ds_test[0]
    #
    # for i in np.random.permutation(len(ds_test))[:1000]:
    #     ds_test[i]
