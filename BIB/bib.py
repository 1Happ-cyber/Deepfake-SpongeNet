import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_, clip_grad_value_
from BIB.src.cbnext import CBNext
from fuse_model import fuse_model



def KL_between_normals(q_distr, p_distr):
    mu_q, sigma_q = q_distr
    mu_p, sigma_p = p_distr
    k = mu_q.size(1)

    mu_diff = mu_p - mu_q
    mu_diff_sq = torch.mul(mu_diff, mu_diff)
    logdet_sigma_q = torch.sum(2 * torch.log(torch.clamp(sigma_q, min=1e-8)), dim=1)
    logdet_sigma_p = torch.sum(2 * torch.log(torch.clamp(sigma_p, min=1e-8)), dim=1)

    fs = torch.sum(torch.div(sigma_q ** 2, sigma_p ** 2), dim=1) + torch.sum(torch.div(mu_diff_sq, sigma_p ** 2), dim=1)
    two_kl = fs - k + logdet_sigma_p - logdet_sigma_q
    return two_kl * 0.5


class BIB(nn.Module):
    def __init__(self, y_dim, backbone = 'BNext-T' ,dimZ=256, beta=1e-3, num_samples=20,num_classes  = 4, use_fuse=False,use_vib = False, use_dyvib=False,
                 use_first=False,use_CBNN = True,freeze_backbone = False,):
        # the dimension of Z
        super().__init__()
        self.beta = beta
        self.dimZ = dimZ
        self.num_samples = num_samples
        self.use_fuse = use_fuse
        self.use_vib = use_vib
        self.use_dyvib = use_dyvib
        self.use_first = use_first

        self.backbone = backbone
        size_map = {"BNext-T": "tiny", "BNext-S": "small", "BNext-M": "middle", "BNext-L": "large"}
        if backbone in size_map:
            size = size_map[backbone]
            self.base_model = None
            # loads the pretrained model

        if use_CBNN:
            # init backbone
            self.base_model = CBNext(num_classes=1000, size=size)
        else:
            print(backbone)


        self.inplanes = self.base_model.fc.in_features  # 1024
        dimZ = self.inplanes // 4


        self.base_model.deactive_last_layer = True
        self.base_model.fc = nn.Identity()

        # eventually freeze the backbone
        self.freeze_backbone = freeze_backbone
        if self.freeze_backbone:
            for p in self.base_model.parameters():
                p.requires_grad = False

        if self.use_vib or self.use_dyvib:#classifier
            # use_first来决定解梯度的位置
            X_dim_half = self.inplanes // 2
            self.encoder = nn.Sequential(
                nn.Linear(in_features=self.inplanes, out_features=X_dim_half),
                nn.ReLU(),
                nn.Linear(in_features=X_dim_half, out_features=X_dim_half),
                nn.ReLU(),
                nn.Linear(in_features=X_dim_half, out_features=2 * self.dimZ))
            if use_fuse:
                self.fuse_model = fuse_model()
            # decoder a simple logistic regression as in the paper
            self.decoder_logits = nn.Linear(in_features=self.dimZ, out_features=y_dim)
            #use_first来区分是最后的VIB还是前面的，进而确定梯度传播的方向
        else:
            # add a new linear layer after the backbone
            self.fc = nn.Linear(self.inplanes, num_classes if num_classes >= 3 else 1)


    def gaussian_noise(self, num_samples, K):
        # works with integers as well as tuples
        return torch.normal(torch.zeros(*num_samples, K), torch.ones(*num_samples, K)).cuda()

    def sample_prior_Z(self, num_samples):
        return self.gaussian_noise(num_samples=num_samples, K=self.dimZ)

    def encoder_result(self, batch):

        encoder_output = self.encoder(batch)

        mu = encoder_output[:, :self.dimZ]
        sigma = torch.nn.functional.softplus(encoder_output[:, self.dimZ:])

        return mu, sigma

    def sample_encoder_Z(self, num_samples, batch):
        batch_size = batch.size()[0]
        mu, sigma = self.encoder_result(batch)

        return mu + sigma * self.gaussian_noise(num_samples=(num_samples, batch_size), K=self.dimZ)

    def forward(self, batch_x, feature_vib=None):
        outs = {}
        features = self.base_model(batch_x)
        if self.use_vib or self.use_dyvib:
            batch_size = features.size()[0]

            if self.use_first:
                features = features.detach()

            # sample from encoder
            encoder_Z_distr = self.encoder_result(features)
            prior_Z_distr = torch.zeros(batch_size, self.dimZ).cuda(), torch.ones(batch_size, self.dimZ).cuda()

            I_ZX_bound = torch.mean(KL_between_normals(encoder_Z_distr, prior_Z_distr))

            to_decoder = self.sample_encoder_Z(num_samples=self.num_samples, batch=features)
            # (num_sample,batchsize,dimZ)
            if self.use_fuse:
                feature_vib = feature_vib.unsqueeze(0).expand(self.num_samples, -1, -1)
                to_decoder = self.fuse_model(to_decoder, feature_vib)
            decoder_logits_mean = torch.mean(self.decoder_logits(to_decoder), dim=0)
            if self.use_dyvib:
                ck =  decoder_logits_mean, I_ZX_bound
            else:
                ck = decoder_logits_mean, I_ZX_bound * self.beta

            outs["logits"] = ck[0]
            return outs, ck[1]
        else:
            outs["logits"] = self.fc(features)
            return outs
