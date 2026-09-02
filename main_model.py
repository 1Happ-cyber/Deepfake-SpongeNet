import gc
import cv2 as cv
import einops
from skimage import feature
import timm
import lightning as L
import numpy as np
from BIB.bib import BIB
from torch import nn
from feature_encoder import HFM
import torch




"Selective Kernel Networks"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#test时注意改参数
class BNext4DFR(L.LightningModule):

    def __init__(self, num_classes=2, backbone='BNext-T', 
                 freeze_backbone=False, add_magnitude_channel=True, add_fft_channel=True, add_lbp_channel=False,
                 learning_rate=1e-4, pos_weight=1., doublebnn=False , use_vib=True, use_fuse=False, use_rgbfreq=False, use_CBNN=False,use_dyvib=False):#用fuse的话是加在bib的encoder后
        #use_vib or use_dyvib only one can be used
        super(BNext4DFR, self).__init__()
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.epoch_outs = []
        self.use_fuse=use_fuse
        self.use_vib=use_vib
        self.use_rgbfreq= use_rgbfreq
        self.use_CBNN = use_CBNN
        self.use_dyvib = use_dyvib


        self.BIB = BIB(
                        y_dim=num_classes if num_classes >= 3 else 1,
                        beta = 1e-4,
                        num_classes = num_classes,
                        backbone = backbone,
                        use_vib = self.use_vib,
                        use_dyvib = self.use_dyvib,
                        use_CBNN = self.use_CBNN,
                        use_first=False,
                        freeze_backbone = freeze_backbone
                       )
        # update the preprocessing metas
        assert isinstance(add_magnitude_channel, bool)
        self.add_magnitude_channel = add_magnitude_channel
        assert isinstance(add_fft_channel, bool)
        self.add_fft_channel = add_fft_channel
        assert isinstance(add_lbp_channel, bool)
        self.add_lbp_channel = add_lbp_channel
        self.new_channels = sum([self.add_magnitude_channel, self.add_fft_channel, self.add_lbp_channel])
        
        # loss parameters
        self.pos_weight = pos_weight
        #self.Model_attention = SKAttention(channel=5,reduction=3)
        if self.new_channels > 0:
            #in_channels=3+self.new_channels
            self.adapter = nn.Conv2d(in_channels=5
                                     , out_channels=3, 
                                     kernel_size=3, stride=1, padding=1)
        else:
            self.adapter = nn.Identity()
            
        # disables the last layer of the backbone
        if self.use_rgbfreq:
            self.combine = HFM(use_dyvib=self.use_dyvib,use_fft=self.add_fft_channel,use_lpb=self.add_lbp_channel,use_mag=self.add_magnitude_channel)
        self.save_hyperparameters()

        
    def forward(self, x):
        x=x.cuda()
        x_rgb = x
        # eventually concat the edge sharpness to the input image in the channel dimension
        if self.add_magnitude_channel or self.add_fft_channel or self.add_lbp_channel:
            x,complement = self.add_new_channels(x)

        if self.use_rgbfreq:
            x_adapted,hsic= self.combine(x_rgb, complement)
            x_adapted = (x_adapted - torch.as_tensor(timm.data.constants.IMAGENET_DEFAULT_MEAN, device=self.device).view(1, -1, 1, 1)) / torch.as_tensor(timm.data.constants.IMAGENET_DEFAULT_STD, device=self.device).view(1, -1, 1, 1)

        else:
            x_adapted = self.adapter(x)
            x_adapted = (x_adapted - torch.as_tensor(timm.data.constants.IMAGENET_DEFAULT_MEAN, device=self.device).view(1, -1, 1, 1)) / torch.as_tensor(timm.data.constants.IMAGENET_DEFAULT_STD, device=self.device).view(1, -1, 1, 1)
            # normalizes the input image

        out = self.BIB(x_adapted)
        if self.use_vib or self.use_dyvib:
            out,ck1 = self.BIB(x_adapted)
            if self.use_rgbfreq:
                return out, ck1, hsic
            else:
                return out,ck1
        else:
            if self.use_rgbfreq:
                return out,hsic
            else:
                return out
 


    
    def configure_optimizers(self):
        #will fill as paper accept
        pass
    
    def _add_new_channels_worker(self, image):
        # convert the image to grayscale
        gray = cv.cvtColor((image.cpu().numpy() * 255).astype(np.uint8), cv.COLOR_BGR2GRAY)
        #(h,w,c)-->gray:(h,w)
        new_channels = []
        if self.add_magnitude_channel:
            new_channels.append(np.sqrt(cv.Sobel(gray,cv.CV_64F,1,0,ksize=7)**2 + cv.Sobel(gray,cv.CV_64F,0,1,ksize=7)**2) )
        #if fast_fourier is required, calculate it
        if self.add_fft_channel:
            image=image.permute(2, 0, 1)
            image=image.unsqueeze(0)
            new_channels.append(20*np.log(np.abs(np.fft.fftshift(np.fft.fft2(gray))) + 1e-9))
        if self.add_lbp_channel:
            new_channels.append(feature.local_binary_pattern(gray, 3, 6, method='uniform'))
        new_channels = np.stack(new_channels, axis=2) / 255
        ck1 = torch.from_numpy(new_channels).to(device).float()
        return ck1 
        
    def add_new_channels(self, images):
        images_copied = einops.rearrange(images, "b c h w -> b h w c")
        new_channels = torch.stack([self._add_new_channels_worker(image) for image in images_copied], dim=0)

        #new_channels:(b,h,w,3个附加层)
        # concatenates the new channels to the input image in the channel dimension
        images_copied = torch.concatenate([images_copied, new_channels], dim=-1)

        new_channels =einops.rearrange(new_channels, "b h w c -> b c h w")
        images_copied = einops.rearrange(images_copied, "b h w c -> b c h w")
        return images_copied,new_channels
    
    def on_train_start(self):
        return self._on_start()
    
    def on_test_start(self):
        return self._on_start()
    
    def on_train_epoch_start(self):
        self._on_epoch_start()
        
    def on_test_epoch_start(self):
        self._on_epoch_start()
        
    def training_step(self, batch, i_batch):
        return self._step(batch, i_batch, phase="train")
    
    def validation_step(self, batch, i_batch):
        return self._step(batch, i_batch, phase="val")
    
    def test_step(self, batch, i_batch):
        return self._step(batch, i_batch, phase="test")
    
    def on_train_epoch_end(self):
        self._on_epoch_end()
        
    def on_test_epoch_end(self):
        self._on_epoch_end()
    def on_validation_epoch_start(self):
        self._on_epoch_start()

    def on_validation_epoch_end(self):
        self._on_epoch_end()

        
    def _step(self, batch, i_batch, phase=None):
        # Removed for paper submission
        # Will be released after AC
        pass
    
    def _on_start(self):
        pass
        # Removed for paper submission
        # Will be released after AC
            
        
    def _on_epoch_start(self):
        self._clear_memory()
        self.epoch_outs = []
    
    def _on_epoch_end(self):
        self._clear_memory()
        # Removed for paper submission
        # Will be released after AC
                    
    def _clear_memory(self):
        gc.collect()
        torch.cuda.empty_cache()
         
        
if __name__ == "__main__":
    net = BNext4DFR(
        num_classes=2,
        use_CBNN=True
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(device)
    net.eval()
    x = torch.randn(8, 3, 224, 224).to(device)

    with torch.no_grad():
        output = net(x)

    print(output[0])


