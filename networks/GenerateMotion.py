import torch
import torch.nn as nn 
from collections import OrderedDict
from einops.layers.torch import Reduce

class Generator(nn.Module): 
    foundation_name = 'motion'
    progressive_name = ['outs', 'prevs']
    def __init__(self, motion_features, decode_dim, repr_dim, total_steps, \
                    num_motions, num_motion_gru, feed_to_next, motion_residual, \
                        overlap, device, args, res_net='iden', excit_ratio=2, vert_channel_expan=2): 
        super().__init__()

        assert len(motion_features) == total_steps, "Check motion feature length and total steps"

        self.motion_features = motion_features
        self.decode_dim = decode_dim
        self.repr_dim = repr_dim
        self.total_steps = total_steps
        self.num_motions = num_motions
        self.total_num_motions = 2 * self.num_motions + 1 
        self.num_motion_gru = num_motion_gru
        self.feed_to_next = feed_to_next
        self.motion_residual = motion_residual
        self.res_net = res_net
        self.overlap = overlap
        self.device = device
        self.cur_step = 0 

        self.motion_grus = nn.ModuleList([nn.GRU(input_size=self.repr_dim, hidden_size=self.decode_dim, num_layers=1, batch_first=True, bidirectional=True)] + 
                                        [nn.GRU(input_size=self.decode_dim, hidden_size=self.decode_dim, num_layers=1, batch_first=True, bidirectional=True) for _ in range(self.num_motion_gru - 1)])
        self.motion_layers = nn.ModuleList([nn.LayerNorm(normalized_shape=(self.total_num_motions, self.decode_dim)) for _ in range(self.num_motion_gru)])
        self.outs = nn.ModuleList(self.get_list_joint_maker())
        self.prev_motion_vectors = None
        
        if self.res_net == 'att' :
            self.avg_pool = nn.AdaptiveAvgPool1d(1)
            self.residual_network = nn.Sequential(
                Reduce('b s c -> b s', reduction='mean'), # global average pooling
                nn.Linear(self.total_num_motions, int(self.total_num_motions//excit_ratio), bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(int(self.total_num_motions//excit_ratio), self.total_num_motions, bias=False),
                nn.Sigmoid(), 
            )
        elif self.res_net == 'hori_concat' : 
            self.residual_network = nn.Sequential(
                nn.Linear(self.repr_dim*2, self.repr_dim),
                nn.LeakyReLU(),
            )
        elif self.res_net == 'vert_concat' : 
            self.residual_network = nn.Sequential(
                nn.Conv1d(self.total_num_motions*2, int(self.total_num_motions*2*vert_channel_expan), 1, 1), 
                nn.LeakyReLU(0.2), 
                nn.Conv1d(int(self.total_num_motions*2*vert_channel_expan), self.total_num_motions, 1, 1), 
                nn.LeakyReLU(0.2), 
            )

        assert self.decode_dim == self.repr_dim, f'In feed_to_next setting, decode_dim({self.decode_dim}) should same as repr_dim({self.repr_dim})'

    def get_list_joint_maker(self): 
        outs = []
        accum_features = 0
        in_features = self.decode_dim
        for feature in self.motion_features :
            accum_features += len(feature)

            out = OrderedDict() 
            
            out['linear1'] = nn.Linear(in_features=in_features, out_features=accum_features * 3)
            out['norm'] = nn.LayerNorm(normalized_shape=accum_features * 3)
            out['act'] = nn.LeakyReLU(0.2)

            out['linear2'] = nn.Linear(in_features=accum_features * 3, out_features=accum_features * 3)

            outs.append(nn.Sequential(out))

        return outs

    def forward(self, latents, cur_step, gen_proc): # latents : [batch_size, cur_step+1,  2 * context + 1, self.repr_dim]
        if self.check_reset_prev_motion(cur_step, latents.shape[0], gen_proc) : 
            self.cur_step = cur_step
            self.prev_motion_vectors = None
        
        motions = []

        if self.prev_motion_vectors is None and self.overlap != 0 and gen_proc : 
            self.prev_motion_vectors = torch.zeros((latents.shape[0], latents.shape[1], self.overlap, latents.shape[-1]), requires_grad=False).to(self.device)

        if self.motion_residual and cur_step > 0 : 
            self.prev_step_feat = torch.zeros((latents.shape[0], cur_step, self.num_motion_gru, latents.shape[-2], self.repr_dim), requires_grad=False).to(self.device)

        for step in range(cur_step + 1) : 
            out = latents[:, step].clone() # [batch_size, 2 * context + 1, self.repr_dim]
            
            if self.feed_to_next and self.overlap != 0 and gen_proc : 
                out[:, :self.overlap] += self.prev_motion_vectors[:, step, :self.overlap]

            for gru_idx, (gru, layer) in enumerate(zip(self.motion_grus, self.motion_layers)) : 
                out, _ = gru(out)
                out = out[:, :, :self.decode_dim] + out[:, :, self.decode_dim:]
                out = layer(out) # [batch_size, 2 * context + 1, self.repr_dim]
                if self.motion_residual and cur_step > 0 :
                    if step != cur_step : 
                        self.prev_step_feat[:, step, gru_idx, :, :] = out.clone().detach()
                    if step > 0 :
                        prev = self.prev_step_feat[:, step-1, gru_idx, :, :].clone()
                        out = self.apply_residual_connection(out, prev)

            if self.feed_to_next and self.overlap != 0 and gen_proc : 
                self.prev_motion_vectors[:, step, :self.overlap] = out[:, -self.overlap:].detach()
            
            out = self.outs[step](out) #(out) # [batch, len(motion_features[:cur_step+1])*3]
            
            step_motion = out.view(out.shape[0], self.total_num_motions, -1, 3) # [batch, total_num_motions, len(motion_features[:cur_step+1]), 3]
            motions.append(step_motion)

        return motions 
    
    def apply_residual_connection(self, out, prev_out): 
        if self.res_net == 'iden' : 
            return out + prev_out 

        elif self.res_net == 'att' :
            res = self.residual_network(prev_out)
            return out + prev_out * res.unsqueeze(2)

        elif self.res_net == 'hori_concat' : 
            cat = torch.cat((out, prev_out), dim=2)
            return self.residual_network(cat)
        
        elif self.res_net == 'vert_concat' : 
            cat = torch.cat((out, prev_out), dim=1) 
            return self.residual_network(cat)

    def check_reset_prev_motion(self, cur_step, batch_size, gen_proc): 
        if gen_proc : 
            if self.prev_motion_vectors is None : 
                return False 
            
            if self.cur_step != cur_step or batch_size != self.prev_motion_vectors.shape[0] : # either step changed or new batch started 
                return True 
        else : 
            return False 