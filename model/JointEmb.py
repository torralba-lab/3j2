import math

from torch import nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch
import torch.nn.functional as nnf
from torchvision.models import resnet


class Identity(nn.Module):
    def forward(self, x): return x


class ImageEncoder(nn.Module):
    def __init__(self, emb_dim, resnet_depth, **kwargs):
        super(ImageEncoder, self).__init__()

        self.conv = getattr(resnet, f'resnet{resnet_depth}')(pretrained=True)
        img_emb_dim = 2048 if resnet_depth >= 50 else 512
        self.conv.fc = Identity()

        self.emb = nn.Sequential(
            nn.BatchNorm1d(img_emb_dim),
            nn.Linear(img_emb_dim, emb_dim),
            nn.Tanh(),
        )

    def forward(self, imgs, ft_cnn=False, **kwargs):
        img_embs = self.conv(imgs) if imgs.ndimension() == 4 else imgs
        if not ft_cnn:
            img_embs = img_embs.detach()
        embs = self.emb(img_embs)
        return embs

class TokenEncoder(nn.Module):
    def __init__(self, word_emb_dim, text_emb_dim, emb_dim,
                 max_seqlen, vocab_size, **kwargs):
        super(TokenEncoder, self).__init__()

        self.wembs = nn.Embedding(vocab_size, word_emb_dim, padding_idx=0)

        k = 3
        p = 0
        s = 1
        l = max_seqlen
        layer_specs = []
        while True:
            lp = (l + 2*p - k) // s + 1
            if lp < 3:
                break
            layer_specs.append(('conv', k, p, s))
            l = lp

            if len(layer_specs) % 4 == 1:
                lp = (l - 4) // 2 + 1
                if lp < 3:
                    break
                layer_specs.append(('pool', k, p, s))
                l = lp
        assert l > 0

        d = word_emb_dim
        d_step = (text_emb_dim - d) // len(layer_specs)
        layers = []
        for i, (t, k, p, s) in enumerate(layer_specs, 1):
            if t == 'conv':
                d_out = d + d_step if i != len(layer_specs) else text_emb_dim
                layers.append(nn.Conv2d(d, d_out, (k, 1), stride=(s, 1), padding=(p, 0)))
                layers.append(nn.BatchNorm2d(d_out))
                layers.append(nn.ReLU(inplace=True))
                d = d_out
            elif t == 'pool':
                layers.append(nn.MaxPool2d((4,1), stride=(2,1)))
        layers.append(nn.AvgPool2d((l, 1)))
        layers.append(nn.Conv2d(d, emb_dim, (1, 1)))
        layers.append(nn.BatchNorm2d(emb_dim))
        layers.append(nn.Tanh())

        self.sembs = nn.Sequential(*layers)

        nparam = sum(map(lambda p: p.data.numel(), self.sembs.parameters()))
        if kwargs.get('debug', False):
            print(self.sembs)
            print(f'text: #layers={len(layer_specs)}, remainder={l}, params={nparam}')

    def forward(self, input):
        wembs = self.wembs(input)
        sembs = self.sembs(wembs.unsqueeze(3).transpose(1, 2)).squeeze(3).squeeze(2)
        return sembs


def _pairwise_jsd(x):
    bsz, dim = x.size()
    xe = x.unsqueeze(0).expand(bsz, bsz, dim)
    lxe = x.log().unsqueeze(0).expand(bsz, bsz, dim)
    lm = ((xe + xe.transpose(0, 1)) / 2).log()
    kl = (xe * (lxe - lm)).sum(2, keepdim=True).squeeze(2)
    jsd = (kl + kl.t()) / 2
    return jsd

def _pairwise_dist(x, y):
    n_x, dim_x = x.size()
    n_y, dim_y = y.size()
    assert dim_x == dim_y
    xe = x.unsqueeze(1).expand(n_x, n_y, dim_x)
    ye = y.unsqueeze(0).expand(n_x, n_y, dim_y)
    return ((xe - ye)**2).sum(2, keepdim=True).squeeze(2)


class JointEmb(nn.Module):
    def __init__(self, **kwargs):
        super(JointEmb, self).__init__()

        self.epoch = 0
        self.ft_after = kwargs['ft_after']

        emb_dim = kwargs['emb_dim']

        self.img_emb = ImageEncoder(**kwargs)
        self.instr_emb = TokenEncoder(**kwargs)
        self.topic_clf = nn.Sequential(
            nn.BatchNorm1d(emb_dim),
            nn.Linear(emb_dim, kwargs['n_topics'] + 1),  # +1 for non-food
            nn.LogSoftmax(),
        )


    def forward(self, imgs, instrs, topics, isfood, epoch=0, ship_embs=True, **kwargs):
        img_embs = self.img_emb(imgs, ft_cnn=epoch > self.ft_after)
        instr_embs = self.instr_emb(instrs)
        topic_preds = self.topic_clf((img_embs + instr_embs) / 2)

        log_topics = topics.log()
        exp_preds = topic_preds.exp()
        topic_loss = nnf.kl_div(topic_preds, topics) + nnf.kl_div(log_topics, exp_preds)

        emb_loss = (((img_embs - instr_embs)**2).mean(1) * isfood).mean()

        no_diag = Variable(1 - torch.eye(len(imgs)).type_as(imgs.data))

        topic_dist = _pairwise_jsd(topics) / math.log(2)
        pdists = [
            _pairwise_dist(instr_embs, instr_embs) / 2,
            _pairwise_dist(img_embs, img_embs) / 2,
            _pairwise_dist(instr_embs, img_embs),
        ]
        pdist_loss = sum(topic_dist * nnf.relu(8 - pd) * no_diag for pd in pdists).mean()

        if ship_embs:
            return emb_loss, topic_loss, pdist_loss, img_embs, instr_embs
        return emb_loss, topic_loss, pdist_loss

    @staticmethod
    def create_inputs():
        return {
            'imgs': torch.FloatTensor(1, 1, 1, 1),
            'instrs': torch.LongTensor(1, 1),
            'topics': torch.FloatTensor(1, 1),
            'isfood': torch.FloatTensor(1),
        }

if __name__ == '__main__':
    batch_size = 10
    opts = {
        'n_topics': 20,
        'max_seqlen': 150,
        'vocab_size': 15000,
        'word_emb_dim': 32,
        'emb_dim': 64,
        'text_emb_dim': 256,
        'resnet_depth': 18,
        'ft_after': 0,
        'debug': True,
    }

    torch.manual_seed(424)
    net = JointEmb(**opts)

    inp = {
        'imgs': Variable(torch.rand(batch_size,  3, 224, 224)),
        'topics': Variable(torch.rand(batch_size, opts['n_topics'] + 1)),
        'instrs': Variable(torch.LongTensor(batch_size, opts['max_seqlen']).random_(opts['vocab_size'])),
        'isfood': Variable(torch.FloatTensor(batch_size).random_(2)),
    }

    outputs = net(**inp)
    loss = sum(outputs[:3])
    print(loss)
    loss.backward()
