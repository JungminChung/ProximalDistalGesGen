import os
import re
import torch 
import librosa
import argparse
import numpy as np 

from tqdm import tqdm

from utils.utils import load_configs, \
                        extract_train_args, \
                        find_last_model_paths, \
                        load_numpy, \
                        get_target_joint_on_step, \
                        normalize_xyz

from utils.audio_tools import get_inference_input

from networks.PRGG import ProgReprGestureGen

from utils.embedding_net import EmbeddingSpaceEvaluator
from utils.BoundedBeatConsistency import evaluate_BC

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--gt_motion_folder', help = 'folder that hold all ground truth motion numpy files')
    parser.add_argument('--audio_folder', help = 'folder containing all audio materail that used on inferencing')

    parser.add_argument('--model_folder_path', help = 'trained model path')

    # bc args 
    parser.add_argument('--gt_multiple', type=int, default=0.7, help = 'The value can be between 0 and 1')
    parser.add_argument('--local_min_threshold', type=int, default=1)
    parser.add_argument('--bc_sigma', type=float, default=0.5)
    parser.add_argument('--bc_mute', type=float, default=1)

    # fgd args 
    parser.add_argument('--eval_net_path', type=str, help='path to autoencoder model for FGD \
                                                            share drive path : https://drive.google.com/file/d/1t_E625IkgbV7a5Otg_hsxQTQM9Y_YzdG/view?usp=sharing')
    parser.add_argument('--gesture_embedding_frame_length', type=str, default=34, choices=[34, 64], help='frame legnth that go through embedding network. pre-trained use 34')
    parser.add_argument('--normalize', action='store_true', help='if given, normalize by mean and std on xyz axis')

    
    return parser.parse_args()

def load_trained_model_and_metas(model_folder_path, device):
    assert os.path.exists(model_folder_path), f"Check your model folder {model_folder_path}"
    
    train_args = extract_train_args(model_folder_path)
    
    config_path = train_args['config']
    configs = load_configs(config_path)

    metas = {**configs, **train_args}
    
    model_path = find_last_model_paths(model_folder_path, inference_all_models=False)[0]
    gen = ProgReprGestureGen(num_wav_features = metas['num_wav_features'] ,
                             motion_features = metas['steps'] ,
                             context = metas['context'] ,
                             repr_dim = metas['repr_dim'] , 
                             decode_dim = metas['decode_dim'] ,
                             total_steps = metas['totalStep'] ,
                             wav_hidden = metas['wav_hidden'] ,
                             num_wav_gru = metas['num_wav_gru'],
                             gru_hidden = metas['gru_hidden'] ,
                             audio_residual = metas['audio_residual'],
                             num_motion_gru = metas['num_motion_gru'],
                             feed_to_next = metas['feed_to_next'], 
                             motion_residual = metas['motion_residual'],
                             overlap = metas['overlap'],
                             res_net = metas['res_net'],
                             excit_ratio = metas['excit_ratio'],
                             vert_channel_expan = metas['vert_channel_expan'],
                             device = device ,
                             args = None).to(device).eval()

    if torch.cuda.is_available() :
        gen.load_state_dict(torch.load(model_path))
    else : 
        gen.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    return gen, metas 

def load_fgd_model(eval_net_path, gesture_embedding_frame_length, device):
    if len(eval_net_path) == 0 : 
        raise ValueError('need for FGD process. can download here : https://github.com/ai4r/Gesture-Generation-from-Trimodal-Context#:~:text=Download-,the%20trained%20models,-and%20extract%20to')
    
    embed_space_evaluator = EmbeddingSpaceEvaluator(None, 4, gesture_embedding_frame_length, eval_net_path, None, device, 1)
    embed_space_evaluator.reset() 

    return embed_space_evaluator

def get_gt_and_audio_pair_in_folder(gt_motion_folder, audio_folder, exceptions=['X.npy', '._']):
    ## motion numpy file and audio wav file must have same numbers
    gt_motions, audios = [], []
    
    gt_np_motions = os.listdir(gt_motion_folder)
    audio_waves = os.listdir(audio_folder)

    for gt_np_motion in gt_np_motions: 
        notTarget = False
        for exc in exceptions: 
            if gt_np_motion.startswith(exc) or gt_np_motion.endswith(exc) : notTarget=True
        if notTarget : continue 

        pair_hind_number = re.findall(r'\d+', gt_np_motion)[0]
        for audio_wav in audio_waves :
            notTarget = False
            isPair = False 
            for exc in exceptions: 
                if exc in audio_wav : notTarget=True
            if notTarget : continue 
            if pair_hind_number in audio_wav : isPair=True 
            if isPair : 
                gt_motions.append(os.path.join(gt_motion_folder, gt_np_motion))
                audios.append(os.path.join(audio_folder, audio_wav))
                break 
    return gt_motions, audios

def inference_one_audio(model, input_wav, metas, quiet=False): 
    target_joint_during_steps = [get_target_joint_on_step(metas['steps'], step) for step in range(metas['totalStep'])]

    total_input_len = len(input_wav[0]) 
    head_len = metas['context']
    tail_len = 2 * metas['context']
    pure_wav_len = total_input_len - head_len - tail_len 
    
    t_start_idx = metas['context']
    t_end_idx = total_input_len - 1
    interval = 2 * metas['context'] - metas['overlap'] + 1
    
    wav_end_idx = head_len + pure_wav_len - 1
    
    progress = range(t_start_idx, t_end_idx,  interval) if quiet else tqdm(range(t_start_idx, t_end_idx,  interval), ncols=100)
    total_steps = metas['totalStep']
    overlap_repo = [[] for _ in range(total_steps)]
    mix_ratios = [i for i in np.arange(1, 0, -1 / (metas['overlap'] + 1))][1:]
    res_gen_motion_npy = None

    for idx, t in enumerate(progress):
        if t - metas['context'] > wav_end_idx : continue 
        if not quiet : 
            progress.set_description(f'Model Inferencing \t ')
        t_wavs_with_context = input_wav[:, t - metas['context']: t + metas['context'] + 1, :]  
        
        _, generateds = model(t_wavs_with_context, total_steps - 1)

        num_gen_motion = 2 * metas['context'] + 1
        for gen_motion_idx in range(num_gen_motion):
            # to cut off head and tail garbage motions from silent audio 
            if idx == 0 and gen_motion_idx < metas['context'] or \
                t+interval > wav_end_idx and t-metas['context']+gen_motion_idx+1 > wav_end_idx :
                continue 
            for step in range(total_steps):
                gen_result = generateds[step][0][
                    gen_motion_idx].detach().cpu().numpy()  # 0 : batch , 0 : current time stamp
                abs_value = gen_result

                abs_motion_vector = np.full((len(metas['idx2jointName']), 3), None)
                abs_motion_vector[target_joint_during_steps[step], :] = abs_value
                abs_motion_vector = abs_motion_vector.astype(np.float32)

                if gen_motion_idx < metas['overlap'] and idx != 0:
                    mix_ratio = mix_ratios[gen_motion_idx]
                    abs_motion_vector = overlap_repo[step][gen_motion_idx] * mix_ratio + abs_motion_vector * (
                            1 - mix_ratio)
                    if gen_motion_idx == metas['overlap'] - 1:
                        overlap_repo[step] = []

                if gen_motion_idx > (num_gen_motion - metas['overlap'] - 1) and (t + interval) < t_end_idx:
                    overlap_repo[step].append(abs_motion_vector)
                    continue

                if step == total_steps - 1:
                    # save final step motion joint results to numpy file
                    if res_gen_motion_npy is None:
                        res_gen_motion_npy = np.expand_dims(abs_motion_vector, 0)
                    else:
                        res_gen_motion_npy = np.append(res_gen_motion_npy,
                                                       np.expand_dims(abs_motion_vector, 0), 0)
    return res_gen_motion_npy
    
def evaluate_FGD(gt, pred, embed_space_evaluator, gesture_embedding_frame_length, device, normalize=False) : 
    if normalize : 
        gt = normalize_xyz(gt)
        pred = normalize_xyz(pred)

    for idx in range(0, len(gt), gesture_embedding_frame_length): 
        start = idx
        end = min(idx + gesture_embedding_frame_length, len(gt))
        
        generated_poses = np.zeros((gesture_embedding_frame_length, gt.shape[1], gt.shape[2]))
        real_poses      = np.zeros((gesture_embedding_frame_length, gt.shape[1], gt.shape[2]))

        generated_poses[:end-start] = pred[start:end, :, :]
        real_poses[:end-start]      = gt[start:end, :, :]

        generated_poses = generated_poses.reshape(gesture_embedding_frame_length, -1)
        real_poses      = real_poses.reshape(gesture_embedding_frame_length, -1)

        generated_poses = torch.from_numpy(np.expand_dims(generated_poses, 0)).float().to(device)
        real_poses      = torch.from_numpy(np.expand_dims(real_poses, 0)).float().to(device)

        embed_space_evaluator.push_samples(None, None, generated_poses, real_poses)
    
    frechet_dist, feat_dist = embed_space_evaluator.get_scores()
    embed_space_evaluator.reset() 
    return frechet_dist, feat_dist

def adjust_len(gt, infer):
    if len(infer)> len(gt):
        infer = infer[:(len(gt))]
        return gt, infer
    elif len(gt)> len(infer):
        gt = gt[:len(infer)]
        return gt, infer
    elif len(gt) == len(infer):
        return gt, infer
    else:
        print('adjust_len_error')

def main(): 
    args = parse_args() 
    print(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model, metas = load_trained_model_and_metas(args.model_folder_path, device) 
    fgd_evaluator = load_fgd_model(args.eval_net_path, args.gesture_embedding_frame_length, device)
    
    gt_motions, audios = get_gt_and_audio_pair_in_folder(args.gt_motion_folder, args.audio_folder) 

    results = {}
    for idx, (gt_motion_path, audio_path) in enumerate(zip(gt_motions, audios)) : 
        print(f'[{idx}/{len(gt_motions)}]', end='\t')
        print('#'*20, 'Motion : ', os.path.basename(gt_motion_path), '& Audio : ', os.path.basename(audio_path), '#'*20)
        gt_motion_numpy = load_numpy(gt_motion_path)
        audio_model_input = get_inference_input(audio_path, metas['num_wav_features'], metas['silence_npy_path'], metas['context']).to(device)
        audio_librosa = librosa.load(audio_path)

        audio_name = os.path.basename(audio_path)

        pr_motion_numpy = inference_one_audio(model, audio_model_input, metas, quiet=True)
        gt_motion_numpy_resized, pr_motion_numpy_resized = adjust_len(gt_motion_numpy, pr_motion_numpy)

        bbc = evaluate_BC(gt_motion_numpy, pr_motion_numpy, audio_librosa, args.gt_multiple, args.local_min_threshold, args.bc_sigma, args.bc_mute, quiet=True) 
        fgd = evaluate_FGD(gt_motion_numpy_resized, pr_motion_numpy_resized, fgd_evaluator, args.gesture_embedding_frame_length, device) 

        results[audio_name] = {
            'BBC_vetrebral' : bbc['vetrebral'],
            'BBC_limb' : bbc['limb'],
            'BBC_hand' : bbc['hand'],
            'fgd_frec_dis' : fgd[0],
            'fgd_feat_dis' : fgd[1],
        }

    print()
    print(f'                  | BBC Ve | BBC Li | BBC Ha | FGD fr | FGD fe')
    print(f'-' * 65)
    BBC_vetrebral, BBC_limb, BBC_hand, fgd_frec_dis, fgd_feat_dis  = 0, 0, 0, 0, 0
    
    for audio_name, evals in results.items(): 
        print(f' [{audio_name}] | {evals["BBC_vetrebral"]:.4f} | {evals["BBC_limb"]:.4f} | {evals["BBC_hand"]:.4f} | {evals["fgd_frec_dis"]:.4f} | {evals["fgd_feat_dis"]:.4f}')
        BBC_vetrebral += evals['BBC_vetrebral']
        BBC_limb += evals['BBC_limb']
        BBC_hand += evals['BBC_hand']
        fgd_frec_dis += evals['fgd_frec_dis']
        fgd_feat_dis += evals['fgd_feat_dis']
    
    BBC_vetrebral = BBC_vetrebral/len(gt_motions)
    BBC_limb = BBC_limb/len(gt_motions)
    BBC_hand = BBC_hand/len(gt_motions)
    fgd_frec_dis = fgd_frec_dis/len(gt_motions)
    fgd_feat_dis = fgd_feat_dis/len(gt_motions)

    print(f'-' * 65)
    print(f'     Average      | {BBC_vetrebral:.4f} | {BBC_limb:.4f} | {BBC_hand:.4f} | {fgd_frec_dis:.4f} | {fgd_feat_dis:.4f}')
    print()

if __name__=='__main__':
    main() 


