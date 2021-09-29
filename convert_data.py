import glob
import os
import pickle
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor

import soundfile
import librosa
from tqdm import tqdm


data_dir = os.path.join('data/lhs_v2', 'train')
files = glob.glob(data_dir + '/**/*.pkl', recursive=True)
wav_dir = 'data/hifigan2/wavs'
clean_dir = 'data/hifigan2/clean'
output_sr = 16000

file_names =  ['_'.join(file[:-4].split('/')[-2:])+'| \n' for file in files]
with open('data/hifigan/training.txt', 'w') as fo:
    fo.writelines(file_names)


def run(sub_files):
    print(len(sub_files))
    for file in tqdm(sub_files):

        with open(file, 'rb') as fo:
            data = pickle.load(fo)
        file_name =  '_'.join(file[:-4].split('/')[-2:])
        wav = data['receiver_audio']
        # new_wav = librosa.resample(wav, 16000, output_sr)
        soundfile.write(os.path.join(wav_dir, file_name+'.wav'), wav, output_sr)

        clean = data['source_audio']
        # new_clean = librosa.resample(clean, 16000, output_sr)
        soundfile.write(os.path.join(clean_dir, file_name+'.wav'), clean, output_sr)


num_process = 80
step = len(files) // num_process + 1
assigned_files = []
for i in range(num_process):
    assigned_files.append(files[step * i: step * (i + 1)])

pool = mp.Pool(processes=num_process)
results = pool.map(run, assigned_files)
# with ThreadPoolExecutor(max_workers=num_process) as executor:
#     executor.map(run, assigned_files)