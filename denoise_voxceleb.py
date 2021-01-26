from tqdm import tqdm
import torchaudio
from pathlib import Path
import json
import os

base_path = '/home/rvirgilli/datasets/vox1_dev_wav'
out_path = '/home/rvirgilli/datasets/vox1_dev_dn'
num_workers = '8'
device = 'cuda:0'

'''
for cls in tqdm(os.listdir(base_path)):
    os.mkdir(os.path.join(out_path, cls))
    for vid in os.listdir(os.path.join(base_path, cls)):
        noisy_dir = os.path.join(base_path, cls, vid)
        out_dir = os.path.join(out_path, cls, vid)
        os.mkdir(out_dir)
        cmd_string = 'python -m denoiser.enhance --master64 --num_workers=' + num_workers + ' --device=' + device +\
                     ' --noisy_dir=' + noisy_dir + ' --out_dir=' + out_dir
        os.system(cmd_string)
        for file in os.listdir(out_dir):
            if '_noisy' in file:
                os.remove(os.path.join(out_dir, file))
            else:
                os.rename(os.path.join(out_dir, file), os.path.join(out_dir, file.replace('_enhanced', '')))
'''
def generate_json(base_path, json_name, exts=[".wav"]):
    audio_files = []
    for root, folders, files in os.walk(base_path, followlinks=True):
        for file in files:
            file = Path(root) / file
            if file.suffix.lower() in exts:
                audio_files.append(str(file.resolve()))
    meta = []
    for idx, file in enumerate(audio_files):
        siginfo, _ = torchaudio.info(file)
        length = siginfo.length // siginfo.channels
        meta.append((file, length))
    with open(json_name, 'w') as outfile:
        json.dump(meta, outfile)

os.path.split()

generate_json(base_path, 'audio_files.json')

cmd_string = 'python -m denoiser.enhance --master64 --num_workers=' + num_workers + ' --device=' + device + \
             ' --noisy_json=audio_files.json --out_dir=' + out_path
os.system(cmd_string)