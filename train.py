import os 
import sys 
sys.path.append(f'{os.path.abspath(".")}')
import copy
import argparse
from tqdm import tqdm
import numpy as np 
import random

import torch
from torch.utils.data import DataLoader 
from torch.utils.tensorboard import SummaryWriter

from utils.utils import get_save_path
from utils.utils import load_configs
from utils.utils import get_target_joint_on_step
from modules.dataset import TrinityDataset
from modules.collator import Collator
from modules.loss import MSEloss
from modules.loss import WGANGP_D_loss
from modules.loss import WGANGP_G_loss
from modules.loss import ContinuityLoss
from modules.loss import BoneLengthLoss
from modules.loss import LossCalculator
from modules.optimize import Custom_Optimizer

from networks.PRGG import ProgReprGestureGen
from networks.Discriminator import GesDiscriminator

def parse_args():
    parser = argparse.ArgumentParser()

    # Environment setting 
    parser.add_argument('--results_path', type=str, default='train_results', help='trained results folder')
    parser.add_argument('--seed', type=int, default=1234, help='random seed')

    # IO setting 
    parser.add_argument('--config', type=str, default='configs/genea_typeG.json', help='path to target config file')
    parser.add_argument('--train_data_folder', type=str, help='train data folder path')
    parser.add_argument('--silence_npy_path', type=str,  help='silence npy file path')
    parser.add_argument('--hierarchy_npy_path', type=str, help='hierarchy npy file path')
    parser.add_argument('--save_itv', type=int, default='10', help='save point on every interval epoch')

    # General Model setting 
    parser.add_argument('--num_wav_features', type=int, default='26', help='How many features we will store for each MFCC vector')
    parser.add_argument('--context', type=int, default=30, choices=[5, 10, 20, 30], 
                                     help='''one way context value added along with input t, total input : context * 2 + 1''')
    parser.add_argument('--overlap', type=int, default=8, help='overlap motion frames b.t.w neighbouring generated results in one direction')
    
    parser.add_argument('--epochs_per_step', nargs='+', type=int, default=[40, 40, 40, 40, 40, 40, 40], help='number of epochs for training on every steps')
    parser.add_argument('--batch_size', type=int, default='24', help='batch size for model')
    
    parser.add_argument('--optim', default='Adam', choices=['Adam', 'SGD'], help='optimizer type. either Adam or SGD')

    parser.add_argument('--gen_lr', type=float, default='0.001', help='learning rate for generator PRGG')
    parser.add_argument('--dis_lr', type=float, default='0.001', help='learning rate for discriminator')
    
    parser.add_argument('--n_critic', type=int, default='1', help='ratio of training generator to discriminator')

    # Audio Encoder setting 
    parser.add_argument('--wav_hidden', type=int, default='128', help='wav feature hidden size in audio encoder')
    parser.add_argument('--gru_hidden', type=int, default='128', help='gru hidden size in audio encoder')
    parser.add_argument('--num_wav_gru', type=int, default='2', help='num of gru layer in audio encoder')
    parser.add_argument('--audio_residual', action='store_true', help='if given, add prev step\'s audio latent to current audio latent like residual connection')
    parser.add_argument('--repr_dim', type=int, default='128', help='representation dimension')

    # Motion Generator setting 
    parser.add_argument('--decode_dim', type=int, default='128', help='output dimension of decoder gru')
    parser.add_argument('--num_motion_gru', type=int, default='2', help='num of gru layer in motion generator')
    parser.add_argument('--feed_to_next', action='store_true', help='feed generated motions to following sequence')
    parser.add_argument('--motion_residual', action='store_true', help='if given, add prev step\'s motion latent to current motion latent like residual connection')
    parser.add_argument('--res_net', default='iden', choices=['iden', 'att', 'hori_concat', 'vert_concat'], help='residual function on motion features')
    parser.add_argument('--excit_ratio', type=float, default='2', help='channel wise shrink ratio on SE block')
    parser.add_argument('--vert_channel_expan', type=float, default='2', help='expansion ration on vertical conv process')

    # Discriminator setting 
    parser.add_argument('--disc_dim', type=int, default='128', help='output dimension of discriminator linear mapper')
    parser.add_argument('--disc_ch', type=int, default='128', help='basic dimension size in discriminator conv')

    # Loss setting 
    parser.add_argument('--lambda_d_loss', type=float, default='1', help='lambda multiply for d loss during train discriminator')
    parser.add_argument('--lambda_gp', type=float, default='1', help='lambda multiply for gradient penalty during train discriminator')
    parser.add_argument('--lambda_g_loss', type=float, default='1', help='lambda multiply for g loss during train PRGG generator')
    parser.add_argument('--lambda_mse', type=float, default='1', help='lambda multiply for mse loss during train PRGG generator')
    parser.add_argument('--lambda_continuity', type=float, default='1', help='lambda multiply for continuity loss during train PRGG generator')
    parser.add_argument('--lambda_bonelength', type=float, default='1', help='lambda multiply for bonelength loss during train PRGG generator')

    return parser.parse_args()

def set_seed(seed): 
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed) # if use multi-GPU
    # torch.backends.cudnn.deterministic = True # could slow-down
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def main(): 
    args = parse_args()
    print(args)
    set_seed(args.seed)

    assert os.path.exists(args.config), f"Check your config file path {args.config}"
    configs = load_configs(args.config)

    motion_features = configs['steps']
    total_steps = configs['totalStep']

    if not os.path.exists(args.results_path) : 
        os.makedirs(args.results_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('target device :', device)

    save_path = get_save_path(args.results_path)
    print(f'Trained model will be saved in {save_path}')

    with open(os.path.join(save_path, 'args.txt'), 'w') as args_file : 
        args_file.write(str(args))
    
    args.epochs_per_step = args.epochs_per_step[:total_steps]
    args.epochs_per_step.extend([args.epochs_per_step[-1]] * (total_steps - len(args.epochs_per_step)))

    writer = SummaryWriter(save_path)

    dataset = TrinityDataset(args.train_data_folder, args.context, args.silence_npy_path, args.context, args.hierarchy_npy_path, configs['num_joints'])
    
    dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size, 
                            shuffle=True, collate_fn=Collator(args.silence_npy_path, args.hierarchy_npy_path).collate_fn)
    
    gen = ProgReprGestureGen(num_wav_features=args.num_wav_features, motion_features=motion_features, 
                            context=args.context, repr_dim=args.repr_dim, 
                            decode_dim=args.decode_dim, total_steps=total_steps, 
                            wav_hidden=args.wav_hidden, num_wav_gru=args.num_wav_gru, 
                            gru_hidden=args.gru_hidden, audio_residual=args.audio_residual, 
                            num_motion_gru=args.num_motion_gru, feed_to_next=args.feed_to_next, motion_residual=args.motion_residual,
                            overlap=args.overlap, res_net = args.res_net, excit_ratio=args.excit_ratio, vert_channel_expan=args.vert_channel_expan, 
                            device=device, args=args).to(device)
    dis = GesDiscriminator(motion_features=motion_features, disc_dim=args.disc_dim, 
                            disc_ch=args.disc_ch, num_motions=args.context, 
                            device=device, args=args).to(device)

    target_joint_during_steps = [get_target_joint_on_step(motion_features, step) for step in range(total_steps)]

    mse = MSEloss()
    wgangp_d_loss = WGANGP_D_loss() 
    wgangp_g_loss = WGANGP_G_loss()
    continuity_loss = ContinuityLoss(overlap=args.overlap)
    bonelength_loss = BoneLengthLoss(bvh_path=os.path.join(args.train_data_folder, '..', os.path.basename(args.hierarchy_npy_path).replace('npy', 'bvh')), 
                                    target_joint_during_steps=target_joint_during_steps, device=device)
    loss_calculator = LossCalculator(mse, wgangp_d_loss, wgangp_g_loss, continuity_loss, bonelength_loss)

    gen_opt = Custom_Optimizer(args.optim, gen, args.gen_lr, args.epochs_per_step)
    dis_opt = Custom_Optimizer(args.optim, dis, args.dis_lr, args.epochs_per_step)
    print(f"[Num of params] G : {gen.count_parameters()}, D : {dis.count_parameters()}")

    run_epoch = 0 
    run_loader = 0 
    for cur_step in range(total_steps): 
        for epoch in range(args.epochs_per_step[cur_step]): 
            gen_opt.adjust_learning_rate(cur_step, run_epoch)
            dis_opt.adjust_learning_rate(cur_step, run_epoch)

            progress_on_epoch = tqdm(dataloader, ncols=100)
            for idx, (load_wavs, load_motions, bvh_paths) in enumerate(progress_on_epoch): 
                    
                progress_on_epoch.set_description(f'Step [{cur_step}] | Epoch : {epoch} / {args.epochs_per_step} | Loader : {idx} / {len(dataloader)} \t')
                wavs = load_wavs.float().to(device) # [batch_size, max_length, mfcc_channel]
                motions = [load_motions[:, :, target_joints, :].float().to(device) 
                                            for idx, target_joints in enumerate(target_joint_during_steps)
                                            if idx < cur_step+1] # list([batch_size, max_length, target_joint, 3])

                prev_motions = None 
                progress_on_bath = tqdm(range(args.context, len(wavs[0]) - args.context, 2*args.context - args.overlap + 1), ncols=100)
                for jdx, t in enumerate(progress_on_bath) : 
                    progress_on_bath.set_description(f'Audio & Motion process : {jdx} / {len(progress_on_bath)} \t ')
                    t_wavs_with_context = wavs[:, t - args.context : t+args.context+1, :] # [batch_size, 2*context+1, mfcc_channel]
                    t_motions = [motion[:, t - args.context : t+args.context+1, :, :] for motion in motions]

                    if prev_motions is None : 
                        prev_motions = [motion[:, :args.overlap, :, :] for motion in motions]

                    # Train discriminator 
                    dis.train()
                    dis_opt.zero_grad() 

                    with torch.no_grad():
                        latents, generateds = gen(t_wavs_with_context, cur_step) # generateds : [batch_size, total_num_motions, len(motion_features[:cur_step+1]), 3]
                    d_reals = dis(t_motions, cur_step)
                    d_fakes = dis(generateds, cur_step)
                    d_loss = loss_calculator.calc_ganD(d_reals, d_fakes)
                    gp = loss_calculator.calc_GP(t_motions, generateds, dis, cur_step, device)

                    final_d_loss = args.lambda_d_loss * d_loss  + args.lambda_gp * gp 
                    final_d_loss.backward()
                    dis_opt.step_optim()

                    loss_calculator.update_MA_d_losses(d_loss, gp, final_d_loss)

                    if idx % args.n_critic == 0 : 
                        # Train generator 
                        gen.train() 
                        gen_opt.zero_grad() 

                        latents, generateds = gen(t_wavs_with_context, cur_step, gen_proc=True)
                        d_fakes = dis(generateds, cur_step)
                        g_loss = loss_calculator.calc_ganG(d_fakes)
                        
                        mse_loss = loss_calculator.calc_mse(generateds, t_motions)
                        cont_loss = loss_calculator.calc_continuity(prev_motions, generateds)
                        bl_loss = loss_calculator.calc_bonelength(generateds, bvh_paths)

                        final_g_loss = args.lambda_g_loss * g_loss + args.lambda_mse * mse_loss + args.lambda_continuity * cont_loss + args.lambda_bonelength * bl_loss
                        final_g_loss.backward() 
                        gen_opt.step_optim()

                        loss_calculator.update_MA_g_losses(g_loss, mse_loss, cont_loss, bl_loss, final_g_loss)
                    
                    prev_motions = [g.detach() for g in generateds]
                
                writer.add_scalar('g_loss', loss_calculator.MA_g_loss.avg, run_loader)
                writer.add_scalar('mse_loss', loss_calculator.MA_mse_loss.avg, run_loader)
                writer.add_scalar('continuity_loss', loss_calculator.MA_continuity_loss.avg, run_loader)
                writer.add_scalar('bonelength_loss', loss_calculator.MA_bonelength_loss.avg, run_loader)
                writer.add_scalar('final_g_loss', loss_calculator.MA_final_g_loss.avg, run_loader)
                writer.add_scalar('d_loss', loss_calculator.MA_d_loss.avg, run_loader)
                writer.add_scalar('gp', loss_calculator.MA_gp.avg, run_loader)
                writer.add_scalar('final_d_loss', loss_calculator.MA_final_d_loss.avg, run_loader)
                
                loss_calculator.reset_losses()
                run_loader += 1 
            
            if (epoch+1) % args.save_itv == 0 or epoch == args.epochs_per_step[cur_step]-1  : 
                torch.save(gen.state_dict(), os.path.join(save_path, f'CS{cur_step}_E' + f'{epoch+1}'.zfill(3) + '_gen.ckpt'))
                torch.save(dis.state_dict(), os.path.join(save_path, f'CS{cur_step}_E' + f'{epoch+1}'.zfill(3) + '_dis.ckpt'))

            run_epoch += 1 

if __name__=='__main__':
    main()