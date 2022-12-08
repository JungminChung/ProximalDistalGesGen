import torch.nn as nn 

from .EncodeAudio import AudioEncoder
from .GenerateMotion import Generator

class ProgReprGestureGen(nn.Module):
    def __init__(self, num_wav_features, motion_features, context, \
                repr_dim, decode_dim, total_steps, wav_hidden, num_wav_gru, \
                    gru_hidden, audio_residual, num_motion_gru, feed_to_next, motion_residual, \
                        overlap, res_net, excit_ratio, vert_channel_expan, device, args):
        super().__init__()
        
        self.E = AudioEncoder(num_wav_features, context, repr_dim, total_steps, wav_hidden, num_wav_gru, gru_hidden, audio_residual, device, args)
        self.G = Generator(motion_features, decode_dim, repr_dim, total_steps, context, num_motion_gru, feed_to_next, motion_residual, overlap, device, args, res_net, excit_ratio, vert_channel_expan)
        
    def forward(self, wav, cur_step, gen_proc=False):
        latents = self.E(wav, cur_step) # [batch_size, cur_step+1, self.repr_dim]
        motions = self.G(latents, cur_step, gen_proc) # list([batch_size, total_num_motions, len(motion_features[:cur_step+1]), 3])
        
        return latents, motions
    
    def count_parameters(self):
        return sum([p.numel() for p in self.parameters() if p.requires_grad])
