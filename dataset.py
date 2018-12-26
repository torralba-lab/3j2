import os
import pickle
import random

from PIL import Image
import numpy as np
import pandas as pd
import torch
import torch.utils.data
import torchvision.transforms as transforms


PAD, UNK, EOS = 0, 1, 2
EOS_TOKS = {'.', '!'}

class RecipeDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, vocab, doctops, img_root, nonfood_ims, vocab_size,
                 max_seqlen, preemb=None, part='train', **kwargs):
        super(RecipeDataset, self).__init__()

        self.img_root = img_root
        self.max_seqlen = max_seqlen

        self.rng = random.Random()
        self.rng.seed(os.urandom(8))

        with open(f'{nonfood_ims}', 'rb') as f_nfi:
            self.nonfood_ims = pickle.load(f_nfi)

        if part == 'train':
            extra_txforms = [
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
            ]
        else:
            extra_txforms = [transforms.CenterCrop(224)]
        txforms = [
            transforms.Scale(256),
            *extra_txforms,
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
        self.transform = transforms.Compose(txforms)

        with open(f'{vocab}', 'rb') as f_vocab:
            vocab = pickle.load(f_vocab)
        self.vocab = ['PAD', 'UNK', '</s>'] + vocab[:vocab_size-3]
        self.w2i = {w: i for i, w in enumerate(self.vocab)}

        self.doctops = pd.read_hdf(f'{doctops}/{part}.h5')

        with open(f'{dataset}_{part}.pkl', 'rb') as f_ds:
            self.data = pickle.load(f_ds)

        self.samples = []
        for i, recipe in enumerate(self.data):
            for j, img in enumerate(recipe.get('images', [])):
                self.samples.append((i, j))
        if part == 'train' and nonfood_ims is not None:
            for i in range(len(self.samples)):
                self.samples.append((None, None))

        if preemb is not None:
            img_emb_dim = 2048 if kwargs['resnet_depth'] >= 50 else 512
            self.preemb = np.memmap(f'{preemb}_{part}.dat', dtype='float32', mode='r',
                                    shape=(len(self.samples), img_emb_dim))

    def __getitem__(self, index):
        r_idx, im_idx = self.samples[index]

        instr_toks = torch.LongTensor(self.max_seqlen).zero_()
        topics = torch.FloatTensor(self.doctops.shape[1] + 1).fill_(1e-10)

        if r_idx is None:
            if hasattr(self, 'preemb'):
                img = torch.from_numpy(self.preemb[index])
            else:
                img = None
                while img is None:
                    try:
                        nfi_path = self.rng.choice(self.nonfood_ims)
                        img = Image.open(nfi_path).convert('RGB')
                    except:
                        img = None
                        pass
                img = self.transform(img)
            topics[0] = 1

            return {
                'imgs': img,
                'instrs': instr_toks,
                'topics': topics,
                'isfood': 0,
            }

        recipe = self.data[r_idx]
        if hasattr(self, 'preemb'):
            img = torch.from_numpy(self.preemb[index])
        else:
            imid = recipe['images'][im_idx]
            img = self.transform(Image.open(os.path.join(self.img_root, imid)))

        topics[1:] = torch.from_numpy(self.doctops.loc[recipe['id']].as_matrix())

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

        return {
            'imgs': img,
            'instrs': instr_toks,
            'topics': topics,
            'isfood': 1,
        }

    def __len__(self):
        return len(self.samples)

    def _decode_instr(self, instr):
        toks = [self.vocab[tok] for tok in instr if tok > 0]
        return ' '.join(toks)

def create(*args, **kwargs):
    return RecipeDataset(*args, **kwargs)

if __name__ == '__main__':
    ds_opts = {
        'dataset': 'data/recipes/recipes',
        'doctops': 'data/topics/ingrs_coo_k20',
        'vocab': 'data/vocab.pkl',
        'nonfood_ims': 'data/nonfood_ims.pkl',
        'img_root': 'data/images',
        'max_seqlen': 150,
        'vocab_size': 15000,
        'preemb': 'data/preemb/img_embs',
        'resnet_depth': 18,
    }
    ds_test = RecipeDataset(**ds_opts, part='test')
    datum = ds_test[0]
    print(ds_test._decode_instr(datum['instrs']))

    # for i in np.random.permutation(len(ds_test))[:1000]:
    #     ds_test[i]
