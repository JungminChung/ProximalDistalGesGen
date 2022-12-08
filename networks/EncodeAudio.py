import torch
import torch.nn as nn 
from collections import OrderedDict

from torch.nn.modules.linear import Identity

class AudioEncoder(nn.Module): 
    foundation_name = 'wav'
    progressive_name = 'steps'
    def __init__(self, num_wav_features, context, repr_dim, total_steps, wav_hidden, num_wav_gru, gru_hidden, audio_residual, device, args): 
        super().__init__()

        self.num_wav_features = num_wav_features
        self.context = context
        self.total_wav_sequence = 2*self.context + 1
        self.repr_dim = repr_dim
        self.wav_hidden = wav_hidden
        self.num_wav_gru = num_wav_gru
        self.gru_hidden = gru_hidden
        self.audio_residual = audio_residual
        self.device = device

        self.wav_linear1 = nn.Linear(in_features=self.num_wav_features, out_features=self.wav_hidden//2)
        self.wav_linear_norm1 = nn.LayerNorm(normalized_shape=self.wav_hidden//2)
        self.wav_linear_act1 = nn.LeakyReLU(0.2)
        self.wav_linear2 = nn.Linear(in_features=self.wav_hidden//2, out_features=self.wav_hidden)
        
        self.wav_grus = nn.ModuleList([nn.GRU(input_size=self.wav_hidden, hidden_size=self.gru_hidden, num_layers=1, batch_first=True, bidirectional=True)] +
                                        [nn.GRU(input_size=self.gru_hidden, hidden_size=self.gru_hidden, num_layers=1, batch_first=True, bidirectional=True) for _ in range(self.num_wav_gru - 1)])
        self.wav_layers = nn.ModuleList([nn.LayerNorm(normalized_shape=(self.total_wav_sequence, self.gru_hidden)) for _ in range(self.num_wav_gru)])

        self.steps = nn.ModuleList(self.get_list_steps(total_steps))
    
    def get_list_steps(self, total_steps): 
        steps = []
        out_features = self.repr_dim
        for _ in range(total_steps):
            in_features = self.gru_hidden
            
            step = OrderedDict() 
            step['linear'] = nn.Linear(in_features=in_features, out_features=out_features)
            step['norm'] = nn.LayerNorm(normalized_shape=out_features)
            step['act'] = nn.LeakyReLU(0.2)
                
            steps.append(nn.Sequential(step))
            
        return steps

    def forward(self, wav, cur_step): # wav : [batch, 2 * context + 1, mfcc channel]
        batch_size = wav.shape[0]
        
        out = self.wav_linear1(wav)
        out = self.wav_linear_norm1(out)
        out = self.wav_linear_act1(out)
        out = self.wav_linear2(out)

        for gru, layer in zip(self.wav_grus, self.wav_layers) : 
            out, _ = gru(out)
            out = out[:, :, :self.gru_hidden] + out[:, :, self.gru_hidden:]
            out = layer(out)

        latents = torch.empty_like(out).to(self.device)
        latents = latents.unsqueeze(0).expand((cur_step+1, ) + out.shape)
        
        for idx in range(cur_step + 1): 
            out = self.steps[idx](out) # [batch, 2 * context + 1, self.repr_dim]
            if self.audio_residual and idx > 0 : 
                out = out + latents[idx-1]
            latents[idx] = out 

        latents = latents.permute(1, 0, 2, 3) # [batch, cur_step+1,  2 * context + 1, self.repr_dim]
        return latents 