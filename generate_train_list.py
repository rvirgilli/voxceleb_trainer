import os, glob
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, help='Path to dataset', required=True)
parser.add_argument('--output', type=str,   default='train_list.txt', help='Name of output file')
args = parser.parse_args()

base_path = args.path
output_file = args.output
classes = os.listdir(base_path)

train_list = ""

for cls in tqdm(classes):
    cls_files = glob.glob(os.path.join(base_path, cls, '*/*.wav'))
    for file in cls_files:
        train_list += cls + ' ' + file.replace(base_path, '').replace('\\', '/')[1:] + '\n'

f = open(output_file, "w")
f.write(train_list)
f.close()