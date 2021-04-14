import glob, os
import math
from tqdm import tqdm
import torchaudio
import pandas as pd
import numpy as np
import argparse
from multiprocessing import Pool

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str,   default='/home/rvirgilli/datasets/vox1_test_short/wav',    help='Dataset path to original dataset')
args = parser.parse_args()

dataset_path = args.path

for file_path in tqdm(glob.iglob(os.path.join(dataset_path, '**/*.wav'), recursive=True)):
    audio, sr = torchaudio.load(file_path)
    duration = audio.shape[1] / sr

    filename = os.path.split(file_path)[1].split('.')[0]
    path = os.path.split(file_path)[0]

    if duration > 6:
        n_seg = math.ceil((duration / 5 + duration / 3) / 2)
        seg_durations = 2 * np.random.rand(n_seg) + 3
        overlap_time = (sum(seg_durations) - duration) / (n_seg - 1)
        start_frame = [0]
        end_frame = []
        seg_audio = []
        for i in range(n_seg):
            start_frame.append(start_frame[i] + int((seg_durations[i] - overlap_time) * sr))
            if i == n_seg - 1:
                end_frame.append(audio.shape[1])
            else:
                end_frame.append(start_frame[i] + int(seg_durations[i] * sr))

            seg_audio = audio[:, start_frame[i]:end_frame[i]]
            seg_file_path = os.path.join(path, filename + '_' + str(i).zfill(2) + '.wav')
            torchaudio.save(seg_file_path, seg_audio, sr)
        os.remove(file_path)