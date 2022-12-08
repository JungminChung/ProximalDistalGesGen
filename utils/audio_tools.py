import os 
import sys
import torch
import numpy as np

import scipy.io.wavfile as wav
from python_speech_features import mfcc

import logging
logging.disable(sys.maxsize)

from .utils import average
from .utils import load_numpy

def get_wav_vector(wav_path, mfcc_channel=26): 
    fs, audio = wav.read(wav_path)
    total_time = audio.shape[0]/fs
    print(f'Audio {os.path.split(wav_path)[-1]} has {(int(total_time)//60)} min {(total_time)%60:.2f} sec audio length.')

    if len(audio.shape) == 2:
        audio = (audio[:, 0] + audio[:, 1]) / 2

    input_vectors = mfcc(audio, winlen=0.02, winstep=0.01, samplerate=fs, numcep=mfcc_channel) # FPS : 1/0.01 -> 100 
    input_vectors = np.transpose([average(input_vectors[:, i], 5) for i in range(mfcc_channel)]) # FPS : 20 
    
    return input_vectors

def save_silence(silence_path, assist_npy_folder): 
    wav_npy = get_wav_vector(silence_path)
    one_wav_npy = np.expand_dims(wav_npy[0], 0) # shape : (1, num_mfcc_channel)
    np.save(os.path.join(assist_npy_folder, 'silence.npy'), one_wav_npy)

def get_inference_input(input_path, num_wav_features, silence_npy_path, context):
    input_wav = get_wav_vector(input_path, num_wav_features)
    print(f'Load input wav file, has length {input_wav.shape[0]}')

    wav_pad = load_numpy(silence_npy_path)
    wav_pad_base_head = np.repeat(wav_pad, repeats=context, axis=0)
    wav_pad_base_tail = np.repeat(wav_pad, repeats=2*context, axis=0)
    
    input_wav = np.append(wav_pad, input_wav, axis=0)
    input_wav = np.append(input_wav, wav_pad_base_head, axis=0)
    input_wav = np.append(input_wav, wav_pad_base_tail, axis=0)

    return torch.tensor(input_wav).unsqueeze(0).float() # [1(batch), 2*context + 1, num_wav_features]