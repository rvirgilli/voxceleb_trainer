import os, glob
from tqdm import tqdm

base_path = 'd:/datasets/vox1_dev/wav'
classes = os.listdir(base_path)

train_list = ""

for cls in tqdm(classes):
    cls_files = glob.glob(os.path.join(base_path, cls, '*/*.wav'))
    for file in cls_files:
        train_list += cls + ' ' + file.replace(base_path, '').replace('\\', '/')[1:] + '\n'

f = open("train_list_vox1.txt", "w")
f.write(train_list)
f.close()