import umap
import random 
import torch 
import torch.nn as nn 
import torch.nn.functional as F

import numpy as np
from scipy import linalg

def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std
        
def ConvNormRelu(in_channels, out_channels, downsample=False, padding=0, batchnorm=True):
    if not downsample:
        k = 3
        s = 1
    else:
        k = 4
        s = 2

    conv_block = nn.Conv1d(in_channels, out_channels, kernel_size=k, stride=s, padding=padding)
    norm_block = nn.BatchNorm1d(out_channels)

    if batchnorm:
        net = nn.Sequential(
            conv_block,
            norm_block,
            nn.LeakyReLU(0.2, True)
        )
    else:
        net = nn.Sequential(
            conv_block,
            nn.LeakyReLU(0.2, True)
        )

    return net

class PoseEncoderConv(nn.Module):
    def __init__(self, length, dim, dim_multiplier=2):
        super().__init__()

        dm = dim_multiplier
        self.net = nn.Sequential(
            ConvNormRelu(dim, 32 * dm, batchnorm=True),
            ConvNormRelu(32 * dm, 64 * dm, batchnorm=True),
            ConvNormRelu(64 * dm, 64 * dm, True, batchnorm=True),
            nn.Conv1d(64 * dm, 32 * dm, 3)
        )

        if length == 64 : 
            self.out_net = nn.Sequential(
                nn.Linear(864 * dm, 256 * dm),  # for 64 frames
                nn.BatchNorm1d(256 * dm),
                nn.LeakyReLU(True),
                nn.Linear(256 * dm, 128 * dm),
                nn.BatchNorm1d(128 * dm),
                nn.LeakyReLU(True),
                nn.Linear(128 * dm, 32 * dm),
            )
        elif length == 34 : 
            self.out_net = nn.Sequential(
                nn.Linear(384 * dm, 256 * dm),  # for 34 frames
                nn.BatchNorm1d(256 * dm),
                nn.LeakyReLU(True),
                nn.Linear(256 * dm, 128 * dm),
                nn.BatchNorm1d(128 * dm),
                nn.LeakyReLU(True),
                nn.Linear(128 * dm, 32 * dm),
            )
        else : 
            raise ValueError('frame legnth should be either 34 or 64')

        self.fc_mu = nn.Linear(32 * dm, 32 * dm)
        self.fc_logvar = nn.Linear(32 * dm, 32 * dm)

    def forward(self, poses, variational_encoding):
        # encode
        poses = poses.transpose(1, 2)  # to (bs, dim, seq)
        out = self.net(poses)
        out = out.flatten(1)
        out = self.out_net(out)

        # return out, None, None
        mu = self.fc_mu(out)
        logvar = self.fc_logvar(out)

        if variational_encoding:
            z = reparameterize(mu, logvar)
        else:
            z = mu
        return z, mu, logvar

class PoseDecoderConv(nn.Module):
    def __init__(self, length, dim, dim_multiplier=2, use_pre_poses=False):
        super().__init__()
        self.use_pre_poses = use_pre_poses

        self.dm = dim_multiplier
        feat_size = 32 * self.dm
        if use_pre_poses:
            self.pre_pose_net = nn.Sequential(
                nn.Linear(dim * 4, 32 * self.dm),
                nn.BatchNorm1d(32 * self.dm),
                nn.ReLU(),
                nn.Linear(32 * self.dm, 32 * self.dm),
            )
            feat_size += 32 * self.dm

        if length == 64:
            self.pre_net = nn.Sequential(
                nn.Linear(feat_size, 128 * self.dm),
                nn.BatchNorm1d(128 * self.dm),
                nn.LeakyReLU(True),
                nn.Linear(128 * self.dm, 256 * self.dm),
            )
        elif length == 34:
            self.pre_net = nn.Sequential(
                nn.Linear(feat_size, 64 * self.dm),
                nn.BatchNorm1d(64 * self.dm),
                nn.LeakyReLU(True),
                nn.Linear(64 * self.dm, 136 * self.dm),
            )
        else:
            assert False

        self.net = nn.Sequential(
            nn.ConvTranspose1d(4 * self.dm, 32 * self.dm, 3),
            nn.BatchNorm1d(32 * self.dm),
            nn.LeakyReLU(0.2, True),
            nn.ConvTranspose1d(32 * self.dm, 32 * self.dm, 3),
            nn.BatchNorm1d(32 * self.dm),
            nn.LeakyReLU(0.2, True),
            nn.Conv1d(32 * self.dm, 32 * self.dm, 3),
            nn.Conv1d(32 * self.dm, dim, 3),
        )

    def forward(self, feat, pre_poses=None):
        if self.use_pre_poses:
            pre_pose_feat = self.pre_pose_net(pre_poses.reshape(pre_poses.shape[0], -1))
            feat = torch.cat((pre_pose_feat, feat), dim=1)

        out = self.pre_net(feat) # bs, 136 * self.dm
        out = out.view(feat.shape[0], 4 * self.dm, -1) # bs, 4 * self.dm, 136/(4 * self.dm)
        out = self.net(out) # bs, pose_dim, seq
        out = out.transpose(1, 2) 
        return out

class EmbeddingNet(nn.Module):
    def __init__(self, args, pose_dim, n_frames, n_words, word_embed_size, word_embeddings, dim_multiplier, mode='pose'):
        super().__init__()
        self.context_encoder = None
        self.pose_encoder = PoseEncoderConv(n_frames, pose_dim, dim_multiplier)
        self.decoder = PoseDecoderConv(n_frames, pose_dim, dim_multiplier)
        self.mode = mode

    def forward(self, in_text, in_audio, pre_poses, poses, input_mode=None, variational_encoding=False):
        if input_mode is None:
            assert self.mode is not None
            input_mode = self.mode

        # context
        if self.context_encoder is not None and in_text is not None and in_audio is not None:
            context_feat, context_mu, context_logvar = self.context_encoder(in_text, in_audio)
            # context_feat = F.normalize(context_feat, p=2, dim=1)
        else:
            context_feat = context_mu = context_logvar = None

        # poses
        if poses is not None:
            poses_feat, pose_mu, pose_logvar = self.pose_encoder(poses, variational_encoding)
            # poses_feat = F.normalize(poses_feat, p=2, dim=1)
        else:
            poses_feat = pose_mu = pose_logvar = None

        # decoder
        if input_mode == 'random':
            input_mode = 'speech' if random.random() > 0.5 else 'pose'

        if input_mode == 'speech':
            latent_feat = context_feat
        elif input_mode == 'pose':
            latent_feat = poses_feat
        else:
            assert False

        out_poses = self.decoder(latent_feat, pre_poses)

        return context_feat, context_mu, context_logvar, poses_feat, pose_mu, pose_logvar, out_poses

    def freeze_pose_nets(self):
        for param in self.pose_encoder.parameters():
            param.requires_grad = False
        for param in self.decoder.parameters():
            param.requires_grad = False

class EmbeddingSpaceEvaluator:
    def __init__(self, args, n_pre_poses, n_poses, embed_net_path, lang_model, device, dim_multiplier=1):
        self.n_pre_poses = n_pre_poses

        # init embed net
        ckpt = torch.load(embed_net_path, map_location=device)
        n_frames = n_poses
        # word_embeddings = lang_model.word_embedding_weights
        mode = 'pose'
        self.pose_dim = ckpt['pose_dim']
        self.net = EmbeddingNet(None, self.pose_dim, n_frames, None, None,
                                None, dim_multiplier, mode).to(device)
        self.net.load_state_dict(ckpt['gen_dict'])
        self.net.train(False)

        # storage
        self.context_feat_list = []
        self.real_feat_list = []
        self.generated_feat_list = []
        self.recon_err_diff = []

    def reset(self):
        self.context_feat_list = []
        self.real_feat_list = []
        self.generated_feat_list = []
        self.recon_err_diff = []

    def get_no_of_samples(self):
        return len(self.real_feat_list)

    def push_samples(self, context_text, context_spec, generated_poses, real_poses):
        # convert poses to latent features
        pre_poses = real_poses[:, 0:self.n_pre_poses]
        context_feat, _, _, real_feat, _, _, real_recon = self.net(None, None, None, real_poses,
                                                                   'pose', variational_encoding=False)
        _, _, _, generated_feat, _, _, generated_recon = self.net(None, None, None, generated_poses,
                                                                  'pose', variational_encoding=False)

        if context_feat:
            self.context_feat_list.append(context_feat.data.cpu().numpy())
        self.real_feat_list.append(real_feat.data.cpu().numpy())
        self.generated_feat_list.append(generated_feat.data.cpu().numpy())

        # reconstruction error
        recon_err_real = F.l1_loss(real_poses, real_recon).item()
        recon_err_fake = F.l1_loss(generated_poses, generated_recon).item()
        self.recon_err_diff.append(recon_err_fake - recon_err_real)

    def get_features_for_viz(self):
        generated_feats = np.vstack(self.generated_feat_list)
        real_feats = np.vstack(self.real_feat_list)

        transformed_feats = umap.UMAP().fit_transform(np.vstack((generated_feats, real_feats)))
        n = int(transformed_feats.shape[0] / 2)
        generated_feats = transformed_feats[0:n, :]
        real_feats = transformed_feats[n:, :]

        return real_feats, generated_feats

    def get_scores(self):
        generated_feats = np.vstack(self.generated_feat_list)
        real_feats = np.vstack(self.real_feat_list)

        def frechet_distance(samples_A, samples_B):
            A_mu = np.mean(samples_A, axis=0)
            A_sigma = np.cov(samples_A, rowvar=False)
            B_mu = np.mean(samples_B, axis=0)
            B_sigma = np.cov(samples_B, rowvar=False)
            try:
                frechet_dist = self.calculate_frechet_distance(A_mu, A_sigma, B_mu, B_sigma)
            except ValueError:
                frechet_dist = 1e+10
            return frechet_dist

        ####################################################################
        # frechet distance
        frechet_dist = frechet_distance(generated_feats, real_feats)

        ####################################################################
        # distance between real and generated samples on the latent feature space
        dists = []
        for i in range(real_feats.shape[0]):
            d = np.sum(np.absolute(real_feats[i] - generated_feats[i]))  # MAE
            dists.append(d)
        feat_dist = np.mean(dists)

        return frechet_dist, feat_dist

    @staticmethod
    def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
        """ from https://github.com/mseitzer/pytorch-fid/blob/master/fid_score.py """
        """Numpy implementation of the Frechet Distance.
        The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
        and X_2 ~ N(mu_2, C_2) is
                d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
        Stable version by Dougal J. Sutherland.
        Params:
        -- mu1   : Numpy array containing the activations of a layer of the
                   inception net (like returned by the function 'get_predictions')
                   for generated samples.
        -- mu2   : The sample mean over activations, precalculated on an
                   representative data set.
        -- sigma1: The covariance matrix over activations for generated samples.
        -- sigma2: The covariance matrix over activations, precalculated on an
                   representative data set.
        Returns:
        --   : The Frechet Distance.
        """

        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert mu1.shape == mu2.shape, \
            'Training and test mean vectors have different lengths'
        assert sigma1.shape == sigma2.shape, \
            'Training and test covariances have different dimensions'

        diff = mu1 - mu2

        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; '
                   'adding %s to diagonal of cov estimates') % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return (diff.dot(diff) + np.trace(sigma1) +
                np.trace(sigma2) - 2 * tr_covmean)




if __name__ == '__main__':
    # for model debugging
    n_frames = 64
    pose_dim = 10
    encoder = PoseEncoderConv(n_frames, pose_dim)
    decoder = PoseDecoderConv(n_frames, pose_dim)

    poses = torch.randn((4, n_frames, pose_dim))
    feat, _, _ = encoder(poses, True)
    recon_poses = decoder(feat)

    print('input', poses.shape)
    print('feat', feat.shape)
    print('output', recon_poses.shape)