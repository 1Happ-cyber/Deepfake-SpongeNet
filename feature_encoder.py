import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2
import hsic
from reweight import CKAChannelReweighting
class MobileNetV2Encoder(nn.Module):
    def __init__(self, in_channels=3, pretrained=True):
        super().__init__()
        assert in_channels in [1, 2, 3], "IN_CHANNEL!!! must be 1 or 3"
        mobilenet = mobilenet_v2(pretrained=pretrained)
        if in_channels == 2:
            original_conv0 = mobilenet.features[0][0]
            new_conv0 = nn.Conv2d(2, original_conv0.out_channels, 
                                  kernel_size=original_conv0.kernel_size,
                                  stride=original_conv0.stride,
                                  padding=original_conv0.padding,
                                  bias=False)
            if pretrained:
                with torch.no_grad():
                    new_conv0.weight[:] = original_conv0.weight.data.mean(dim=1, keepdim=True)
            mobilenet.features[0][0] = new_conv0
            
        if in_channels == 1:
            original_conv0 = mobilenet.features[0][0]
            new_conv0 = nn.Conv2d(1, original_conv0.out_channels, 
                                  kernel_size=original_conv0.kernel_size,
                                  stride=original_conv0.stride,
                                  padding=original_conv0.padding,
                                  bias=False)

            if pretrained:
                with torch.no_grad():
                    new_conv0.weight[:] = original_conv0.weight.data.mean(dim=1, keepdim=True)
            mobilenet.features[0][0] = new_conv0

        self.encoder = nn.Sequential(*list(mobilenet.features)[:14])
        self.out_channels = 96

    def forward(self, x):
        return self.encoder(x)
            

from combine import RgbFreqFusion 

class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels=3):
        super().__init__()
        self.decoder_layers = nn.Sequential(
            self._make_upsample_block(in_channels, 128),     # H/8
            self._make_upsample_block(128, 64),         # H/4
            self._make_upsample_block(64, 32),          # H/2
            self._make_upsample_block(32, 16),          # H
        )
        self.final_conv = nn.Conv2d(16, out_channels, kernel_size=1, stride=1, padding=0)
        self.tanh = nn.Tanh()

    def _make_upsample_block(self, in_c, out_c):
        return nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.decoder_layers(x)
        x = self.final_conv(x)
        return self.tanh(x)

class HFM(nn.Module):
    def __init__(self, pretrained=True, use_dyvib=False,use_fft= True,use_lpb = True,use_mag = True):
        super().__init__()
        self.use_dyvib=use_dyvib
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.pool1 = nn.AdaptiveAvgPool2d(1)
        self.pool2 = nn.AdaptiveAvgPool2d(1)
        self.rew = CKAChannelReweighting()
        self.rgb_encoder = MobileNetV2Encoder(in_channels=3, pretrained=pretrained)
        in_channels = 0
        if use_fft: in_channels += 1
        if use_lpb: in_channels += 1
        if use_mag: in_channels += 1
        self.freq_lbp_encoder = MobileNetV2Encoder(in_channels=in_channels, pretrained=pretrained)
        
        fusion_channels = self.rgb_encoder.out_channels
        self.fusion_module = RgbFreqFusion(channels=fusion_channels,reduction=16)
        self.decoder = Decoder(in_channels=fusion_channels * 2, out_channels=3)

    def forward(self, x_rgb, x_fft_lbp):
        # x_freq_lbp = get_fft_spectrum(x_rgb)
        x_freq_lbp = x_fft_lbp
        # x_freq_lbp = x_rgb
        f_rgb = self.rgb_encoder(x_rgb)
        f_freq_lbp = self.freq_lbp_encoder(x_freq_lbp)
        # with torch.no_grad():
        #     f1 = self.pool1(f_rgb)
        #     f2 = self.pool1(f_freq_lbp)
        #     f1 = torch.flatten(f1, 1)
        #     f2 = torch.flatten(f2, 1)
        #     loss_hsic = hsic.cka_to_loss_weight(hsic.CKA_precise(f1, f2))
        f_tilde_rgb, f_tilde_freq_lbp = self.fusion_module(f_rgb, f_freq_lbp)
        # f_tilde_rgb, f_tilde_freq_lbp = self.rew(f_tilde_rgb, f_tilde_freq_lbp).to('cuda')
        # print(f_tilde_rgb.)
        # with torch.no_grad():
            
        f1 = self.pool1(f_tilde_rgb)
        f2 = self.pool2(f_tilde_freq_lbp)
        f1 = torch.flatten(f1, 1)
        f2 = torch.flatten(f2, 1)
        loss_hsic = hsic.hsic_normalized(f1,f2)
        
        # loss_hsic +=loss_hsic1
        f_cat = torch.cat([f_tilde_rgb, f_tilde_freq_lbp], dim=1)
        enhancement_map = self.decoder(f_cat)

        output_image = x_rgb + enhancement_map
        return output_image, loss_hsic