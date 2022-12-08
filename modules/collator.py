import torch 
import numpy as np

from utils.utils import load_numpy

class Collator():
    def __init__(self, silence_npy_path, bvh_npy_path):
        self.silence_npy_path = silence_npy_path
        self.bvh_npy_path = bvh_npy_path

    def collate_fn(self, batch):
        wavs, xyzs, bvh_paths = zip(*batch)
        wavs, xyzs, bvh_paths = list(wavs), list(xyzs), list(bvh_paths)

        if len(wavs) != 1 :
            wav_pad_base = load_numpy(self.silence_npy_path)
            xyz_pad_base = load_numpy(self.bvh_npy_path)
            xyz_pad_base = xyz_pad_base.reshape((xyz_pad_base.shape[0], xyzs[0].shape[1], xyzs[0].shape[2]))
            
            lengths = []
            for wav in wavs:
                lengths.append(len(wav))
            max_length = max(lengths)

            for idx in range(len(wavs)) : 
                num_path = max_length - len(wavs[idx])
                
                wav_pad = np.repeat(wav_pad_base, num_path, axis=0)
                wavs[idx] = np.append(wavs[idx], wav_pad, axis=0)

                xyz_pad = np.repeat(xyz_pad_base, num_path, axis=0)
                xyzs[idx] = np.append(xyzs[idx], xyz_pad, axis=0)

        # wavs : [batch_size, max_length, 2 * context + 1, mfcc_channel]
        # xyzs : [batch_size, max_length, total_joint, 3]
        # bvh_paths : list(path_to_bvh), length : batch_size
        return (torch.tensor(wavs), torch.tensor(xyzs), bvh_paths)