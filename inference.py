from distutils.core import run_setup
import os
from typing import Type 
import cv2 
import sys 
sys.path.append(f'{os.path.abspath(".")}')
import torch
import argparse

import numpy as np 
from tqdm import tqdm

from utils.utils import load_numpy
from utils.utils import make_filters
from utils.utils import load_configs
from utils.utils import combine_video_and_audio
from utils.utils import extract_train_args
from utils.utils import find_last_model_paths
from utils.utils import get_target_joint_on_step
from utils.audio_tools import get_inference_input

from networks.PRGG import ProgReprGestureGen

from utils.motion_tools import get_bvh_vector
from utils.motion_tools import create_hierarchy_nodes
from utils.motion_tools import nodes_to_child_list
from utils.motion_tools import read_bvh
from utils.motion_tools import make_image_from_abs_childlist
from utils.motion_tools import rot_vec_to_abs_pos_vec
from utils.motion_tools import rearrange_joint_2dto1d

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_wav', type=str, help='path to input wav file')
    parser.add_argument('--input_motion', type=str, default='', help='path to GT bvh or npy file correspond to input wav file')
    
    parser.add_argument('--trained_model_folder', type=str, help='results folder that hold trained model checkpoints')
    parser.add_argument('--inference_all_models', action='store_true', help='if given, inference all model in trained model folder. Not given, only infer last step trained model')
    parser.add_argument('--hierarchy_bvh_path', type=str, default='dataset/hierarchyMean.bvh', help='hierarchy bvh file path')

    parser.add_argument('--filter_mincutoff_lambda', type=float, default=0.06)
    parser.add_argument('--filter_beta_lambda', type=float, default=0.1)

    return parser.parse_args()

def main(): 
    args = parse_args()
    print(args)

    assert os.path.exists(args.input_wav), f"Check your input wav file path {args.input_wav}"
    assert os.path.exists(args.trained_model_folder), f"Check your model folder {args.trained_model_folder}"
    
    train_args = extract_train_args(args.trained_model_folder)
    
    config_path = train_args['config']
    configs = load_configs(config_path)

    motion_features = configs['steps']
    total_steps = configs['totalStep']
    total_num_joint = len(configs['idx2jointName'])

    model_paths = find_last_model_paths(args.trained_model_folder, args.inference_all_models)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('target device :', device)

    input_wavs = get_inference_input(args.input_wav, 
                                     train_args['num_wav_features'], 
                                     train_args['silence_npy_path'], 
                                     train_args['context']).to(device)

    gen = ProgReprGestureGen(num_wav_features = train_args['num_wav_features'] ,
                             motion_features = motion_features ,
                             context = train_args['context'] ,
                             repr_dim = train_args['repr_dim'] , 
                             decode_dim = train_args['decode_dim'] ,
                             total_steps = total_steps ,
                             wav_hidden = train_args['wav_hidden'] ,
                             num_wav_gru = train_args['num_wav_gru'],
                             gru_hidden = train_args['gru_hidden'] ,
                             audio_residual = train_args['audio_residual'],
                             num_motion_gru = train_args['num_motion_gru'],
                             feed_to_next = train_args['feed_to_next'], 
                             motion_residual = train_args['motion_residual'],
                             overlap = train_args['overlap'],
                             res_net = train_args['res_net'],
                             excit_ratio = train_args['excit_ratio'],
                             vert_channel_expan = train_args['vert_channel_expan'],
                             device = device ,
                             args = args).to(device).eval()
    hierarchy = read_bvh(args.hierarchy_bvh_path)
    nodes = create_hierarchy_nodes(hierarchy)
    child_list = nodes_to_child_list(nodes)

    given_gt = False 
    if args.input_motion != '':
        given_gt = True
        
        if args.input_motion[-3:] == 'bvh': 
            gt = get_bvh_vector(args.input_motion)
        elif args.input_motion[-3:] == 'npy': 
            gt = load_numpy(args.input_motion)
            if gt.shape[-1] != 3 : 
                gt = np.reshape(gt, (-1, configs['num_joints'], 3)) 

        input_wav_no_pad_size = input_wavs[:, train_args['context']:input_wavs.shape[1]-train_args['context'], :].shape[1]
        min_length = min(gt.shape[0], input_wav_no_pad_size)
        gt = gt[:min_length]

    target_joint_during_steps = [get_target_joint_on_step(motion_features, step) for step in range(total_steps)]
    for model_idx, model_path in enumerate(model_paths): 
        if torch.cuda.is_available() :
            gen.load_state_dict(torch.load(model_path))
        else : 
            gen.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
 
        t0 = input_wavs.shape[1]
        x0 = load_numpy(train_args['hierarchy_npy_path']).reshape(configs['num_joints'], 3) # [57, 3]
        min_cutoff = np.ones_like(x0) * args.filter_mincutoff_lambda
        beta = np.ones_like(x0) * args.filter_beta_lambda
        
        filters = make_filters(total_steps, t0 ,x0 ,min_cutoff ,beta)
            
        len_movies = total_steps+1 if given_gt else total_steps
        file_name = f'infer_{os.path.splitext(os.path.basename(model_path))[0]}_{os.path.basename(args.input_wav).split(".")[0]}.mp4'
        results_video_file_path = os.path.join(args.trained_model_folder, file_name)
        video_writer = cv2.VideoWriter(results_video_file_path, 
                                    cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),
                                    20.0, (512 * len_movies, 512))

        model_name = model_path.split(os.sep)[-1]
        
        total_input_len = len(input_wavs[0]) 
        head_len = train_args['context']
        tail_len = 2 * train_args['context']
        pure_wav_len = total_input_len - head_len - tail_len 
        
        t_start_idx = train_args['context']
        t_end_idx = total_input_len - 1
        interval = 2 * train_args['context'] - train_args['overlap'] + 1
        
        wav_end_idx = head_len + pure_wav_len - 1
        
        progress = tqdm(range(t_start_idx, t_end_idx,  interval), ncols=100)
        make_frames = 0 
        overlap_repo = [[] for _ in range(total_steps)] 
        mix_ratios = [i for i in np.arange(1, 0, -1/(train_args['overlap']+1))][1:]
        res_gen_motion_npy = None 
        for idx, t in enumerate(progress) : 
            if t - train_args['context'] > wav_end_idx : continue 
            progress.set_description(f'Model : {model_name} \t ')
            t_wavs_with_context = input_wavs[:, t - train_args['context'] : t+train_args['context']+1, :] # [batch_size, 2*context+1, mfcc_channel]

            _, generateds = gen(t_wavs_with_context, total_steps - 1)

            num_gen_motion = 2 * train_args['context'] + 1 
            for gen_motion_idx in range(num_gen_motion):
                frame = None 
                ready_to_draw = False 
                # to cut off head and tail garbage motions from silent audio 
                if idx == 0 and gen_motion_idx < train_args['context'] or \
                    t+interval > wav_end_idx and t-train_args['context']+gen_motion_idx+1 > wav_end_idx :
                    continue 
                for step in range(total_steps):
                    gen_result = generateds[step][0][gen_motion_idx].detach().cpu().numpy() # 0 : batch , 0 : current time stamp
                    abs_value = gen_result  
                    
                    abs_motion_vector = np.full((total_num_joint, 3), None)
                    abs_motion_vector[target_joint_during_steps[step], :] = abs_value
                    abs_motion_vector = abs_motion_vector.astype(np.float32) 
                    
                    if gen_motion_idx < train_args['overlap'] and idx != 0 :
                        mix_ratio = mix_ratios[gen_motion_idx]
                        abs_motion_vector = overlap_repo[step][gen_motion_idx] * mix_ratio + abs_motion_vector * (1-mix_ratio)
                        if gen_motion_idx == train_args['overlap'] - 1 : 
                            overlap_repo[step] = []

                    if gen_motion_idx > (num_gen_motion - train_args['overlap'] - 1) and (t + interval) < t_end_idx : 
                        overlap_repo[step].append(abs_motion_vector)
                        continue
                    
                    abs_motion_vector = filters[step](make_frames, abs_motion_vector)
                    
                    if frame is None : 
                        frame = make_image_from_abs_childlist(abs_motion_vector, child_list)
                        if total_steps == 1 : ready_to_draw = True # if only one step in total, no process to go in 'else' path 
                        
                    else : 
                        frame = cv2.hconcat([frame, make_image_from_abs_childlist(abs_motion_vector, child_list)])
                        if step == total_steps - 1 : 
                            if given_gt :  
                                if make_frames >= len(gt) : 
                                    make_frames = min(make_frames, len(gt)-1) # for mismatch btw gt motion length and audio length 
                                frame = cv2.hconcat([make_image_from_abs_childlist(gt[make_frames], child_list), frame])
                            ready_to_draw = True

                    if ready_to_draw : 
                        video_writer.write(frame)
                        make_frames += 1 

                        # save final step motion joint results to numpy file  
                        if res_gen_motion_npy is None : 
                            res_gen_motion_npy = np.expand_dims(abs_motion_vector, 0)
                        else : 
                            res_gen_motion_npy = np.append(res_gen_motion_npy, 
                                                        np.expand_dims(abs_motion_vector, 0), 0)

        npy_file_name = f'infer_{os.path.splitext(os.path.basename(model_path))[0]}_{os.path.basename(args.input_wav).split(".")[0]}.npy'
        npy_save_path = os.path.join(args.trained_model_folder, npy_file_name)
        np.save(npy_save_path, res_gen_motion_npy)

        video_writer.release()
        print('Combining audio to video... ', end= ' ')
        combine_video_and_audio(results_video_file_path, args.input_wav)
        print(f'Done!!')

if __name__=='__main__':
    main()