import os
from cv2 import repeat 
import numpy as np 

from torch.utils.data.dataset import Dataset
from utils.utils import load_numpy

class TrinityDataset(Dataset): 
    def __init__(self, data_folder, context, silence_npy_path, num_motions, hierarchy_npy_path, num_joint) : 
        self.data_folder = data_folder
        npys = os.listdir(self.data_folder) 

        self.X_npys = [] 
        self.Y_npys = [] 

        for npy in npys :
            if npy.endswith('X.npy'):
                self.X_npys.append(load_numpy(os.path.join(data_folder, npy)))
                self.Y_npys.append(load_numpy(os.path.join(data_folder, npy[:-5]+'Y.npy')))
        
        assert len(self.X_npys) + len(self.Y_npys) == len(npys), f"The number of X and Y file mis-matched in {self.data_folder}, check file names."
        self.X_npys = np.array(self.X_npys)
        self.Y_npys = np.array(self.Y_npys)

        self.wav_pad_base = load_numpy(silence_npy_path)
        self.wav_pad_base = np.repeat(self.wav_pad_base, repeats=context, axis=0)
        
        self.bvh_pad_base = load_numpy(hierarchy_npy_path)
        self.bvh_pad_base = self.bvh_pad_base.reshape(1, num_joint, 3)
        self.bvh_pad_base = np.repeat(self.bvh_pad_base, repeats=num_motions, axis=0)

    def __len__(self) :
        return len(self.X_npys) 

    def __getitem__(self, index) :
        wav = self.X_npys[index]
        wav = np.append(self.wav_pad_base, wav, axis=0)
        wav = np.append(wav, self.wav_pad_base, axis=0)

        xyz = self.Y_npys[index]
        xyz = np.append(self.bvh_pad_base, xyz, axis=0)
        xyz = np.append(xyz, self.bvh_pad_base, axis=0)

        return wav, xyz, None 