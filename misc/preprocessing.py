import os 
import sys 
sys.path.append(f'{os.path.abspath(".")}')
import glob
import argparse

from utils.motion_tools import save_hierarchy, get_bvh_vector
from utils.audio_tools import save_silence, get_wav_vector
from utils.utils import size_match, save_wav_bvh_vectors, accumulate_bvhs, get_mean_by_axis

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_folder', type=str, help='folder that contains wav and bvh folders')
    parser.add_argument('--save_folder', type=str, default='dataset', help='target save folder that will hold preprocessed data')
    parser.add_argument('--train', action='store_true', help='if given, regared as train dataset')

    parser.add_argument('--context', type=int, default=30, help='context frame in one direction')
    parser.add_argument('--overlap', type=int, default=8, help='overlap frame b.t.w neighbouring data in one direction')
    
    # audio args 
    parser.add_argument('--mfcc_inputs', type=int, default=26, help='How many features we will store for each MFCC vector')

    # gesture args 
    parser.add_argument('--n_joint', type=int, default=57, help='''Number of joint on motion data. 
                                                                   Output of motion vector size will be determined to n_joint * 3 (x, y, z)''')
    parser.add_argument('--silence_path', type=str, default='misc/silence.wav', help='path to silence.wav file') 

    return parser.parse_args()

def main():
    args = parse_args()

    assert os.path.exists(args.data_folder), f'Check your dataset folder {args.data_folder}'
    assert os.path.exists(args.silence_path), f'Check your silence.wav file path.. {args.silence_path}'
    
    all_files  = glob.glob(os.path.join(args.data_folder, '**'), recursive=True)

    wavs, bvhs = [], [] 
    for f in all_files : 
        if f.endswith('wav') : wavs.append(f)
        if f.endswith('bvh') : bvhs.append(f)

    assert len(wavs) == len(bvhs), f'bvh and wav file number unmatched. num wav : {len(wavs)}, num bvh : {len(bvhs)}'
    
    dataset_save_folder = os.path.join(args.save_folder, 'train') if args.train \
                            else os.path.join(args.save_folder, 'test')
    if not os.path.exists(dataset_save_folder) : os.makedirs(dataset_save_folder)


    name_wav_bvh = [] 
    for wav in wavs: 
        base_file_name = os.path.split(wav)[-1].replace('.wav', '')
        bvh = [b for b in bvhs if base_file_name in b][0]
        name_wav_bvh.append([base_file_name, wav, bvh])

    for idx, (base_name, wav, bvh) in enumerate(name_wav_bvh): 
        wav_vector = get_wav_vector(wav, args.mfcc_inputs)
        bvh_vector = get_bvh_vector(bvh)
        wav_vector, bvh_vector = size_match(wav_vector, bvh_vector)
        # if args.train : 
        #     accum_bvh = accumulate_bvhs(accum_bvh, bvh_vector)

        print(f' {idx+1} / {len(name_wav_bvh)} \t ', end = '')
        save_wav_bvh_vectors(base_name, wav_vector, bvh_vector, dataset_save_folder)
    
    save_hierarchy(bvh, args.save_folder)
    save_silence(args.silence_path, args.save_folder)
    # if args.train : 
    #     save_hierarchy(get_mean_by_axis(accum_bvh), args.save_folder, name='hierarchyMean')

if __name__=='__main__':
    main()