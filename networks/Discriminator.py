import torch.nn as nn 
from collections import OrderedDict

class GesDiscriminator(nn.Module): 
    foundation_name = 'disc'
    progressive_name = 'mappers'
    def __init__(self, motion_features, disc_dim, disc_ch, num_motions, device, args): 
        super().__init__()
        self.motion_features = motion_features
        self.disc_dim = disc_dim
        self.disc_ch = disc_ch
        self.total_num_motions = 2 * num_motions + 1 
        self.device = device

        self.mappers = nn.ModuleList(self.get_list_disc_mapper())
        
        if self.total_num_motions == 11 : 
            self.disc_convs = nn.Sequential(
                nn.AdaptiveAvgPool1d(self.total_num_motions),
                nn.Conv1d(in_channels=self.disc_dim, out_channels=self.disc_ch*1, kernel_size=4, stride=1),
                nn.LeakyReLU(0.2),
                nn.LayerNorm(normalized_shape=(self.disc_ch*1, 8)),
                nn.Conv1d(in_channels=self.disc_ch*1, out_channels=self.disc_ch*4, kernel_size=4, stride=2),
                nn.LeakyReLU(0.2),
                nn.LayerNorm(normalized_shape=(self.disc_ch*4, 3)),
                nn.Conv1d(in_channels=self.disc_ch*4, out_channels=self.disc_ch*8, kernel_size=3, stride=1),
                nn.LeakyReLU(0.2),
                nn.LayerNorm(normalized_shape=(self.disc_ch*8, 1)),
            )

        elif self.total_num_motions == 21 : 
            self.disc_convs = nn.Sequential(
                nn.AdaptiveAvgPool1d(self.total_num_motions),
                nn.Conv1d(in_channels=self.disc_dim, out_channels=self.disc_ch*1, kernel_size=4, stride=1),
                nn.LeakyReLU(0.2),
                nn.LayerNorm(normalized_shape=(self.disc_ch*1, 18)),
                nn.Conv1d(in_channels=self.disc_ch*1, out_channels=self.disc_ch*2, kernel_size=4, stride=2),
                nn.LeakyReLU(0.2),
                nn.LayerNorm(normalized_shape=(self.disc_ch*2, 8)),
                nn.Conv1d(in_channels=self.disc_ch*2, out_channels=self.disc_ch*4, kernel_size=4, stride=2),
                nn.LeakyReLU(0.2),
                nn.LayerNorm(normalized_shape=(self.disc_ch*4, 3)),
                nn.Conv1d(in_channels=self.disc_ch*4, out_channels=self.disc_ch*8, kernel_size=3, stride=1),
                nn.LeakyReLU(0.2),
                nn.LayerNorm(normalized_shape=(self.disc_ch*8, 1)),
            )
        
        elif self.total_num_motions == 41 : 
            self.disc_convs = nn.Sequential(
                nn.AdaptiveAvgPool1d(self.total_num_motions),
                nn.Conv1d(in_channels=self.disc_dim, out_channels=self.disc_ch*1, kernel_size=4, stride=1),
                nn.LeakyReLU(0.2),
                nn.LayerNorm(normalized_shape=(self.disc_ch*1, 38)),
                nn.Conv1d(in_channels=self.disc_ch*1, out_channels=self.disc_ch*2, kernel_size=4, stride=2),
                nn.LeakyReLU(0.2),
                nn.LayerNorm(normalized_shape=(self.disc_ch*2, 18)),
                nn.Conv1d(in_channels=self.disc_ch*2, out_channels=self.disc_ch*2, kernel_size=4, stride=2),
                nn.LeakyReLU(0.2),
                nn.LayerNorm(normalized_shape=(self.disc_ch*2, 8)),
                nn.Conv1d(in_channels=self.disc_ch*2, out_channels=self.disc_ch*4, kernel_size=4, stride=2),
                nn.LeakyReLU(0.2),
                nn.LayerNorm(normalized_shape=(self.disc_ch*4, 3)),
                nn.Conv1d(in_channels=self.disc_ch*4, out_channels=self.disc_ch*8, kernel_size=3, stride=1),
                nn.LeakyReLU(0.2),
                nn.LayerNorm(normalized_shape=(self.disc_ch*8, 1)),
            )
        
        elif self.total_num_motions == 61 : 
            self.disc_convs = nn.Sequential(
                nn.AdaptiveAvgPool1d(self.total_num_motions),
                nn.Conv1d(in_channels=self.disc_dim, out_channels=self.disc_ch*1, kernel_size=4, stride=1),
                nn.LeakyReLU(0.2),
                nn.LayerNorm(normalized_shape=(self.disc_ch*1, 58)),
                nn.Conv1d(in_channels=self.disc_ch*1, out_channels=self.disc_ch*2, kernel_size=4, stride=2),
                nn.LeakyReLU(0.2),
                nn.LayerNorm(normalized_shape=(self.disc_ch*2, 28)),
                nn.Conv1d(in_channels=self.disc_ch*2, out_channels=self.disc_ch*4, kernel_size=4, stride=2),
                nn.LeakyReLU(0.2),
                nn.LayerNorm(normalized_shape=(self.disc_ch*4, 13)),
                nn.Conv1d(in_channels=self.disc_dim*4, out_channels=self.disc_ch*4, kernel_size=4, stride=1),
                nn.LeakyReLU(0.2),
                nn.LayerNorm(normalized_shape=(self.disc_ch*4, 10)),
                nn.Conv1d(in_channels=self.disc_dim*4, out_channels=self.disc_ch*8, kernel_size=4, stride=2),
                nn.LeakyReLU(0.2),
                nn.LayerNorm(normalized_shape=(self.disc_ch*8, 4)),
                nn.Conv1d(in_channels=self.disc_ch*8, out_channels=self.disc_ch*8, kernel_size=4, stride=1),
                nn.LeakyReLU(0.2),
                nn.LayerNorm(normalized_shape=(self.disc_ch*8, 1)),
            )
        
        else : 
            raise ValueError(f'generated num motioin should one of 11, 21, 41, 61. but given {self.total_num_motions}')

        self.disc_out = nn.Sequential(
            nn.Linear(in_features=self.disc_ch*8, out_features=self.disc_ch*4),
            nn.LayerNorm(normalized_shape=self.disc_ch*4),
            nn.LeakyReLU(0.2),
            nn.Linear(in_features=self.disc_ch*4, out_features=self.disc_ch//2),
            nn.LayerNorm(normalized_shape=self.disc_ch//2),
            nn.LeakyReLU(0.2),
            nn.Linear(in_features=self.disc_ch//2, out_features=1),
        )

    def get_list_disc_mapper(self): 
        mappers = []
        accum_features = 0
        for feature in self.motion_features :
            accum_features += len(feature)
            mapper = OrderedDict() 

            mapper['linear1'] = nn.Linear(in_features = accum_features*3, out_features=self.disc_dim//2)
            mapper['norm1'] = nn.LayerNorm(normalized_shape=self.disc_dim//2)
            mapper['act1'] = nn.LeakyReLU(0.2)
            mapper['linear2'] = nn.Linear(in_features = self.disc_dim//2, out_features=self.disc_dim)
            mapper['norm2'] = nn.LayerNorm(normalized_shape=self.disc_dim)
            mapper['act2'] = nn.LeakyReLU(0.2)
                
            mappers.append(nn.Sequential(mapper))
        return mappers

    def forward(self, motions, cur_step): # motions : list([batch, total_num_motions, num_joint, 3])
        assert len(motions) == (cur_step + 1), 'generated motion and cur_step value mis-matched'

        batch_size = motions[0].shape[0]

        critics = [] 
        for step, motion in enumerate(motions) : # motion : [batch, total_num_motions, num_joint, 3]
            motion = motion.view(batch_size, self.total_num_motions, -1)
            
            out = self.mappers[step](motion) # [batch, total_num_motions, disc_dim] 
            out = out.permute(0, 2, 1) # [batch, disc_dim, total_num_motions] 
            
            out = self.disc_convs(out)
            out = out.squeeze(2)
            out = self.disc_out(out)

            critics.append(out)

        return critics

    def count_parameters(self):
        return sum([p.numel() for p in self.parameters() if p.requires_grad])