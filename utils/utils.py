import os 
import ast
import json
import torch 
import random
import shutil
import numpy as np 
import moviepy.editor as mpe
from .oneEuroFilter import OneEuroFilter

def accumulate_bvhs(pre_bvh, cur_bvh):
    if pre_bvh is None:
        return cur_bvh
    else:
        return np.concatenate((pre_bvh, cur_bvh), axis=0)

def get_mean_by_axis(arr, axis=0):
    return np.mean(arr, axis=axis)

def load_numpy(path):
    return np.load(path)

def get_save_path(results_path):
    idx = max([int(f.split(f'_')[-1]) for f in os.listdir(results_path) if f.startswith('train')]+[0]) + 1 
    path = os.path.join(os.path.abspath(results_path), f'train_{idx}')
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def average(arr, n): # Replace every "n" values by their average
    end = n * int(len(arr)/n)
    return np.mean(arr[:end].reshape(-1, n), 1)

def size_match(vector1, vector2):
    min_len = min(len(vector1), len(vector2))

    vector1 = vector1[:min_len]
    vector2 = vector2[:min_len]

    return vector1, vector2

def save_wav_bvh_vectors(base_name, wav_vector, bvh_vector, target_folder): 
    np.save(os.path.join(target_folder, f'{base_name}_X.npy'), wav_vector)
    np.save(os.path.join(target_folder, f'{base_name}_Y.npy'), bvh_vector)
    print(f'{base_name} file saved... wav shape : {wav_vector.shape}, bvh shape : {bvh_vector.shape}')
    print()

def load_configs(config_path):
    with open(config_path, 'r') as f :
        data = json.load(f)
    return data

def combine_video_and_audio(video_path, audio_path):
    print(f'add audio {audio_path} to video {video_path}...')
    origin_vid_path = video_path
    backup_vid_path = os.path.join(os.path.dirname(origin_vid_path), 
                                    '.back'.join(os.path.splitext(os.path.basename(origin_vid_path))))
    shutil.copy(origin_vid_path, backup_vid_path)

    my_clip = mpe.VideoFileClip(backup_vid_path)
    audio_background = mpe.AudioFileClip(audio_path)
    final_audio = mpe.CompositeAudioClip([audio_background])
    final_clip = my_clip.set_audio(final_audio)
    final_clip.write_videofile(origin_vid_path, audio_codec="aac")

    os.remove(backup_vid_path)

def get_target_joint_on_step(motion_features, step):
    return [joint_num for i in range(step+1) for joint_num in motion_features[i]]

def extract_train_args(model_folder_path): 
    args_txt_path = os.path.join(model_folder_path, 'args.txt')
    with open(args_txt_path, 'r') as f: 
        content = f.read()
    
    content = content.replace('Namespace(', '')
    content = content.replace("'", '')
    content = content[:-1]
    
    key_value = [] 
    for c in content.split('='): 
        args = c.split(',')
        if len(args) == 1 : 
                key_value.append(args[0])
        elif len(args) == 2 : 
                key_value.append(args[0])
                key_value.append(args[1])
        elif len(args) > 2 : 
                key_value.append(', '.join(args[:-1]))
                key_value.append(args[-1])
    
    args = {k.strip():convert_type(v.strip()) for k, v in zip(key_value[::2], key_value[1::2])}
    return args 


def convert_type(value):
    try : 
        res = int(value)
        return res 
    except : 
        try : 
            res = ast.literal_eval(value)
            return res
        except : 
            return value 

def find_last_model_paths(model_folder_path, inference_all_models):
    gens = [model for model in os.listdir(model_folder_path) if 'gen' in model and model.endswith('ckpt') and not model.startswith('._')]
    
    if len(gens) == 1 : 
        print(f'find 1 model')
        return [os.path.join(model_folder_path, gens[0])]

    max_step = max([int(gen[2]) for gen in gens])

    last_model_paths = []
    for step in range(max_step+1):
        gens_by_step = [gen for gen in gens if f'CS{step}' in gen]
        max_by_step = max([int(gen[5:8]) for gen in gens_by_step])
        last_model_paths.append(os.path.join(model_folder_path, f'CS{step}_E'+f'{max_by_step}'.zfill(3)+'_gen.ckpt'))
    
    if inference_all_models: 
        print(f'find {max_step+1} models')
        return last_model_paths

    else : 
        print(f'inference model step : {max_step}')
        return [path for path in last_model_paths if f'CS{max_step}_E' in path]

def make_filters(num_filters, t0 ,x0 ,min_cutoff ,beta): 
    return [OneEuroFilter(t0 ,x0 ,min_cutoff ,beta) for _ in range(num_filters)]

def normalize_xyz(np_vectors):
    # np_vector : [batch, joints, xyz]
    results = np.zeros_like(np_vectors)
    for axis in range(np_vectors.shape[2]): 
        mean = np.mean(np_vectors[:, :, axis])
        std = np.std(np_vectors[:, :, axis]) + 1e-10
        results[:, :, axis] = ( np_vectors[:, :, axis] - mean ) / std 
    return results

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)