from tqdm import tqdm
import pandas as pd
import numpy as np
import os, glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, help='Path to dataset', required=True)
parser.add_argument('--output', type=str,   default='train_list.txt', help='Name of output file')
parser.add_argument('--seed', type=str, default=42, help='numpy random seed')
parser.add_argument('--mult', type=int, default=4, help='times each file will be listed for same class and diff class')
parser.add_argument('--nref', type=int, default=1, help='number of reference audios for each probe audio')
args = parser.parse_args()

rng = np.random.default_rng(args.seed)
#TODO Fix random number generator with fixed seed

prior_veri_test = ''
dataset_path = args.path
if prior_veri_test == '':
    files = list(set(glob.iglob(os.path.join(dataset_path, '**/*.wav'), recursive=True)))
    files = [file.replace(dataset_path + '/', '') for file in files]
    files = list(set(files))
else:
    df = pd.read_csv('veri_test.txt', sep=' ', header=None, names=['label', 'verif0', 'probe'])
    files = list(set(list(df['verif0']) + list(df['probe'])))

with open('files.txt', 'w') as f:
    f.write('\n'.join(files))

files_by_class = {}
for file in files:
    file = file.replace(os.path.join(dataset_path, ''), '')
    cls = file.split('/')[0]
    if cls not in files_by_class.keys():
        files_by_class[cls] = []

    files_by_class[cls].append(file)

files.sort()

mult = args.mult
n_ref = args.nref
veri_test = ""
for file in tqdm(files):
    file = file.replace(os.path.join(dataset_path, ''), '')
    cls = file.split('/')[0]
    for _ in range(mult):
        ### same class test
        class_list = files_by_class[cls].copy()
        class_list.remove(file)
        same_class_list = rng.choice(class_list, size=n_ref, replace=False)
        veri_test += '1 ' + file + ' ' + ' '.join(same_class_list) + '\n'

        ### different class test
        test = '0 ' + file + ' '
        if n_ref > 1:
            class_list = files_by_class[cls].copy()
            class_list.remove(file)
            same_class_list = rng.choice(class_list, size=n_ref - 1, replace=False)
            test += ' '.join(same_class_list)

        diff_classes = list(files_by_class.keys())
        diff_classes.remove(cls)
        diff_class = rng.choice(diff_classes)
        diff_file = rng.choice(files_by_class[diff_class])
        test += ' ' + diff_file

        veri_test += test + '\n'

with open(args.output, "w") as text_file:
    text_file.write(veri_test)