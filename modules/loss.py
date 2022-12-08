import os 
import torch 
import torch.nn as nn 
import numpy as np 

from torch.autograd import grad 
from torch.autograd import Variable

from utils.movingAverage import MovingAverage

from utils.motion_tools import read_bvh
from utils.motion_tools import create_hierarchy_nodes

class ContinuityLoss(nn.Module):
    def __init__(self, overlap):
        super(ContinuityLoss, self).__init__()
        self.overlap = overlap 
        self.sml1 = nn.SmoothL1Loss(reduction='none')

    def forward(self, prev_motion, cur_motion): # both size : [batch_size, total_num_motions, len(motion_features[:cur_step+1]), 3] 
        if self.overlap == 0 : 
            return torch.tensor(0.0)
        loss = self.sml1(prev_motion[:, -1 * self.overlap:], cur_motion[:, :self.overlap]) 
        loss = loss.sum(dim=3).sum(dim=2).sum(dim=1) # sum over xyz & joints & time steps 
        loss = loss.mean() # average over batch 
        return loss

class BoneLengthLoss(nn.Module):
    def __init__(self, bvh_path, target_joint_during_steps, device):
        super(BoneLengthLoss, self).__init__()
        self.target_joint_during_steps = target_joint_during_steps
        self.device = device
        self.bvh_path = bvh_path

        self.offset_dicts = {}
        self.node_dicts = {} 
        self.criterion = nn.MSELoss()
        
    def forward(self, generated, bvh_path, step_idx): 
        loss = 0 
        if bvh_path is None : # trinity dataset 
            if 'trinity' not in self.offset_dicts.keys():
                node = create_hierarchy_nodes(read_bvh(self.bvh_path))
                joint_offsets = self.get_joint_offsets_from_node(node)
                self.node_dicts['trinity'] = node
                self.offset_dicts['trinity'] = joint_offsets
            
            node = self.node_dicts['trinity']
            joint_offsets = self.offset_dicts['trinity']
            
        else : # ted dataset 
            bvh_file_name = os.path.basename(bvh_path).split('.bvh')[0]
            if bvh_file_name not in self.offset_dicts.keys() : 
                node = create_hierarchy_nodes(read_bvh(bvh_path))
                joint_offsets = self.get_joint_offsets_from_node(node)
                self.node_dicts[bvh_file_name] = node
                self.offset_dicts[bvh_file_name] = joint_offsets
                
            node = self.node_dicts[bvh_file_name]
            joint_offsets = self.offset_dicts[bvh_file_name]
            
        target_joint = self.target_joint_during_steps[step_idx]
        for parent_idx in range(len(node)): 
            if parent_idx not in target_joint : continue 
            gen_joint_idx_parent = target_joint.index(parent_idx)
            children_idx_list = node[parent_idx]['children']
            for children_idx in children_idx_list: 
                if children_idx not in target_joint : continue 
                gen_joint_idx_child = target_joint.index(children_idx)
                
                gen_offsets = torch.square(generated[:, :, [gen_joint_idx_parent], :] - generated[:, :, [gen_joint_idx_child], :])
                gen_offsets = torch.sum(gen_offsets, len(generated.shape)-1)
                gen_offsets = torch.sqrt(gen_offsets)
                
                target_offset = joint_offsets[parent_idx, children_idx]

                loss += self.criterion(gen_offsets, target_offset)
        return loss.float()

    def get_joint_offsets_from_bvh(self, bvh_path):
        node = create_hierarchy_nodes(read_bvh(bvh_path))
        joint_offsets = self.get_joint_offsets_from_node(node)
        return joint_offsets
        
    def get_joint_offsets_from_node(self, node): 
        joint_offsets = np.zeros((len(node), len(node)))
        for idx, n in enumerate(node): 
            children = n['children'] 
            for child_idx in children : 
                abs_offset = np.sqrt(np.sum(np.square(node[child_idx]['offset'])))
                joint_offsets[idx, child_idx] = abs_offset
        joint_offsets = torch.from_numpy(joint_offsets).float().to(self.device)
        return joint_offsets

class MSEloss(nn.Module):
    def __init__(self):
        super(MSEloss, self).__init__()
        self.mse = nn.MSELoss()
    
    def forward(self, generated, target):
        return self.mse(generated, target)

class WGANGP_G_loss(nn.Module):
    def __init__(self):
        super(WGANGP_G_loss, self).__init__()

    def forward(self, D_fake):
        g_loss = -torch.mean(D_fake)

        return g_loss

class WGANGP_D_loss(nn.Module):
    def __init__(self):
        super(WGANGP_D_loss, self).__init__()

    def forward(self, D_real, D_fake):
        d_loss = -torch.mean(D_real)+torch.mean(D_fake)

        return d_loss

    def gradient_penalty(self, real_motions, fake_motions, dis, cur_step, device):
        x_hats = []
        for real_motion, fake_motion in zip(real_motions, fake_motions):
            epsilon = torch.rand(real_motion.size(0), 1, 1, 1)
            epsilon = epsilon.expand_as(real_motion).to(device)

            x_hat = Variable((epsilon*real_motion) + ((1-epsilon)*fake_motion), requires_grad=True).to(device)
            x_hats.append(x_hat)
        
        D_x_hats = dis(x_hats, cur_step)
        
        gradient_penalties = 0 
        for idx, D_x_hat in enumerate(D_x_hats): 
            grad_outputs = torch.ones(D_x_hat.size()).to(device) 

            # computing gradient 
            gradient = grad(
                outputs = D_x_hat,
                inputs = x_hats[idx],
                grad_outputs = grad_outputs,
                create_graph =True,
                retain_graph=True,
            )[0]

            gradient = gradient.view(real_motions[idx].size(0), -1)
            gradient_norm = torch.sqrt(torch.sum(gradient ** 2, dim=1) + 1e-5)
            gradient_penalty = (gradient_norm ** 2).mean() # zero centered gradient penalty
            gradient_penalties += gradient_penalty

        return gradient_penalties

class LossCalculator(object):
    def __init__(self, mse_loss, gan_d_loss, gan_g_loss, continuity_loss, bonelength_loss, gp=True):
        self.mse_loss = mse_loss
        self.gan_d_loss = gan_d_loss 
        self.gan_g_loss = gan_g_loss
        self.continuity_loss = continuity_loss
        self.bonelength_loss = bonelength_loss
        self.gp = gp

        self.MA_d_loss = MovingAverage()
        self.MA_gp = MovingAverage()
        self.MA_final_d_loss = MovingAverage()

        self.MA_g_loss = MovingAverage()
        self.MA_mse_loss = MovingAverage()
        self.MA_continuity_loss = MovingAverage()
        self.MA_bonelength_loss = MovingAverage()
        self.MA_final_g_loss = MovingAverage()
    
    def calc_continuity(self, prev_motions, cur_motions):
        continuity_loss = 0 
        for prev_motion, cur_motion in zip(prev_motions, cur_motions): 
            continuity_loss += self.continuity_loss(prev_motion, cur_motion)
        return continuity_loss

    def calc_bonelength(self, generateds, bvh_paths):
        bonelength_loss = 0 
        for idx, (generated, bvh_path) in enumerate(zip(generateds, bvh_paths)): 
            bonelength_loss += self.bonelength_loss(generated, bvh_path, idx)
        return bonelength_loss

    def calc_mse(self, generateds, t_motions):
        mse_losses = 0 
        for generated, t_motion in zip(generateds, t_motions):
            mse_losses += self.mse_loss(generated, t_motion)
        return mse_losses 

    def calc_ganD(self, d_reals, d_fakes):
        d_losses = 0 
        for d_real, d_fake in zip(d_reals, d_fakes): 
            d_losses += self.gan_d_loss(d_real, d_fake)
        return d_losses

    def calc_ganG(self, d_fakes):
        d_losses = 0 
        for d_fake in d_fakes: 
            d_losses += self.gan_g_loss(d_fake)
        return d_losses 
    
    def calc_GP(self, t_motions, generateds, dis, cur_step, device):
        gps = self.gan_d_loss.gradient_penalty(t_motions, generateds, dis, cur_step, device)
        return gps
    
    def update_MA_d_losses(self, d_loss, gp, final_d_loss):
        self.MA_d_loss.add_item(d_loss.item())
        self.MA_gp.add_item(gp.item())
        self.MA_final_d_loss.add_item(final_d_loss.item())

    def update_MA_g_losses(self, g_loss, mse_loss, cont_loss, bl_loss, final_g_loss):
        self.MA_g_loss.add_item(g_loss.item())
        self.MA_mse_loss.add_item(mse_loss.item())
        self.MA_final_g_loss.add_item(final_g_loss.item())
        self.MA_continuity_loss.add_item(cont_loss.item())
        self.MA_bonelength_loss.add_item(bl_loss.item())
    
    def reset_losses(self):
        self.MA_d_loss.reset()
        self.MA_gp.reset()
        self.MA_final_d_loss.reset()
        self.MA_g_loss.reset()
        self.MA_mse_loss.reset()
        self.MA_final_g_loss.reset()
        self.MA_continuity_loss.reset()
        self.MA_bonelength_loss.reset()
