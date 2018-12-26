from collections import OrderedDict
import argparse
import html
import os
import pickle
import re
import sys
import unicodedata
import urllib.parse

from joblib import Parallel, delayed
from tqdm import tqdm
import spacy

PROJ_ROOT = os.path.abspath(os.path.dirname(__file__))
DATA_ROOT = os.path.join(PROJ_ROOT, 'data')
sys.path.append(PROJ_ROOT)

REPLACEMENTS = {
    u'\x91':"'", u'\x92':"'", u'\x93':'"', u'\x94':'"', u'\xa9':'',
    u'\xba': ' degrees ', u'\xbc':' 1/4', u'\xbd':' 1/2', u'\xbe':' 3/4',
    u'\xd7':'x', u'\xae': '',
    '\\u00bd':' 1/2', '\\u00bc':' 1/4', '\\u00be':' 3/4',
    u'\\u2153':' 1/3', '\\u00bd':' 1/2', '\\u00bc':' 1/4', '\\u00be':' 3/4',
    '\\u2154':' 2/3', '\\u215b':' 1/8', '\\u215c':' 3/8', '\\u215d':' 5/8',
    '\\u215e':' 7/8', '\\u2155':' 1/5', '\\u2156':' 2/5', '\\u2157':' 3/5',
    '\\u2158':' 4/5', '\\u2159':' 1/6', '\\u215a':' 5/6', '\\u2014':'-',
    '\\u0131':'1', '\\u2122':'', '\\u2019':"'", '\\u2013':'-', '\\u2044':'/',
    '\\u201c':'\\"', '\\u2018':"'", '\\u201d':'\\"', '\\u2033': '\\"',
    '\\u2026': '...', '\\u2022': '', '\\u2028': ' ', '\\u02da': ' degrees ',
    '\\uf04a': '', u'\xb0': ' degrees ', '\\u0301': '', '\\u2070': ' degrees ',
    '\\u0302': '', '\\uf0b0': ''
}

RE_NT = re.compile(r'\\[nt]') # remove over-escaped line breaks and tabs
RE_FRAC_L = re.compile(r'\b([^\d\s]+)/(.*)\b') # split non-fractions
RE_FRAC_R = re.compile(r'\b(.*)/([^\d\s]+)\b') # e.g. 350 deg/gas mark
RE_WS = re.compile(r'\s+') # remove extra whitespace
RE_EDASH = re.compile(r' -- ')

def prepro_txt(text):
    text = html.unescape(text)

    for unichar, replacement in REPLACEMENTS.items():
        text = text.replace(unichar, replacement)
    text = unicodedata.normalize('NFKD', text)

    try:
        text = urllib.parse.unquote(text)
    except UnicodeDecodeError:
        pass # if there's an errant %, unquoting will yield an invalid char

    # some extra tokenization
    text = ' - '.join(text.split('-'))
    text = ' & '.join(text.split('&'))
    text = ' . '.join(text.split(';'))


    text = RE_EDASH.sub(' -- ', text)
    text = RE_NT.sub(' ', text)
    text = RE_FRAC_L.sub(r'\1 / \2', text)
    text = RE_FRAC_R.sub(r'\1 / \2', text)
    text = RE_WS.sub(' ', text)

    return text.strip().lower()

nlp = spacy.load('en')

def fmt_recipe(recipe, verbose=0):
    fmt_instrs = []
    for instr in recipe['instructions']:
        for sent in nlp(prepro_txt(instr)).sents:
            fmt_instrs.append([tok.lower_ for tok in sent])
    recipe['instructions'] = fmt_instrs

    del recipe['partition']

    return recipe

if __name__ != '__main__':
    exit()

#======================================================================================
parser = argparse.ArgumentParser()
parser.add_argument('part', choices=('train', 'val', 'test'))
parser.add_argument('--ds',
                    default='/data/vision/torralba/health-habits/im2recipe/recipe1M')
parser.add_argument('--jobs', '-j', default=-1, type=int)
parser.add_argument('--verbose', '-v', default=0, action='count')
args = parser.parse_args()
#======================================================================================

layers_cache = os.path.join(DATA_ROOT, 'layers_cache.pkl')
if os.path.isfile(layers_cache):
    if args.verbose:
        print('Loading from cache.')
    with open(layers_cache, 'rb') as f_recipes:
        recipes = pickle.load(f_recipes)
else:
    import ujson as json

    recipes_by_id = {}

    with open(os.path.join(DATA_ROOT, 'blacklist.txt')) as f_bl:
        blacklist = {l.rstrip() for l in f_bl}

    if args.verbose:
        print('Loading layer 1')
    with open(os.path.join(args.ds, 'layer1.json')) as f_l1:
        for i, recipe in enumerate(json.load(f_l1)):
            assert recipe['instructions']
            recipes_by_id[recipe['id']] = recipe

    if args.verbose:
        print('Loading layer 2')
    with open(os.path.join(args.ds, 'layer2.json')) as f_l2:
        for recipe_imgs in json.load(f_l2):
            recipe = recipes_by_id[recipe_imgs['id']]
            recipe.update(recipe_imgs)

    recipes = []
    for r in recipes_by_id.values():
        datum = {
            'id': r['id'],
            'title': r['title'],
            'partition': r['partition'],
            'instructions': [instr['text'] for instr in r['instructions']],
        }
        imids = [im['id'] for im in r.get('images', []) if im['id'] not in blacklist]
        if imids:
            datum['images'] = imids
        recipes.append(datum)

    with open(layers_cache, 'wb') as f_recipes:
        pickle.dump(recipes, f_recipes)

with Parallel(n_jobs=args.jobs, verbose=args.verbose) as workers:
    if args.verbose:
        print(f'Formatting {args.part} recipes.')

    part_recipes = [recipe for recipe in recipes if recipe['partition'] == args.part]
    recipes_fmt = workers(delayed(fmt_recipe)(r, args.verbose) for r in part_recipes)

    if args.verbose:
        print(f'Writing out {args.part} data.')
    with open(os.path.join(DATA_ROOT, f'recipes_{args.part}.pkl'), 'wb') as f_ds:
        pickle.dump(recipes_fmt, f_ds)
