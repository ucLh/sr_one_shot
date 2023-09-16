from abc import ABC
from .config import cfg_pixel, cfg_perceptual
import torch
from torchvision.models.vgg import make_layers
from enum import Enum


class TunerTypes(Enum):
    """
    Enum with all tuner types
    """
    PerceptualLossTuner = 'perceptual'
    PixelLossTuner = 'pixel'


class AbstractTuner(ABC):
    """
    Abstract Tuner class that set the tune method interface
    """
    def tune(self, model, hr_t, lr_t, n_iters):
        pass


class PixelLossTuner(AbstractTuner):
    """
    Pixel loss tuner. Performs gradient steps on set super resolution (SR) model using MSE loss.
    MSE loss is applied to SR network output and a normalized tensor of the original high resolution image.
    This approach of MSE loss usage is usually called `pixel loss`, hence the name.
    """
    def __init__(self, model, device=None):
        self.cfg = cfg_pixel
        self.pixel_loss = torch.nn.MSELoss(reduction='mean')
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.cfg.lr)

    def __repr__(self):
        return 'PixelLossTuner'

    def tune(self, model, hr_t, lr_t, n_iters=1):
        """
        Tunes model inplace
        """
        model.train()
        for i in range(n_iters):
            sr_t = model(lr_t)
            loss = self.pixel_loss(sr_t, hr_t)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            del sr_t


class PerceptualLossTuner(AbstractTuner):
    """
    Perceptual loss tuner. Performs gradient steps on set super resolution (SR) model using MSE loss.
    To compute the loss we take SR network output (sr_t) and a normalized tensor of the original
    high resolution image (hr_t).
    We then infer (sr_t) and (hr_t) through the first few layers of a frozen network pretrained on image net (vgg in
    this case) to obtain tensors (sr_feat) and (hr_feat). MSE loss is then applied to (sr_feat) and (hr_feat), and
    a gradient step is done.
     This approach of MSE loss usage is usually called `perceptual loss`, hence the name.
    """
    def __init__(self, model, device):
        self.cfg = cfg_perceptual
        self.content_loss = torch.nn.MSELoss(reduction='mean')
        self.device = device
        self.extractor = self.get_feature_extractor()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.cfg.lr)

    def __repr__(self):
        return 'PerceptualLossTuner'

    def get_feature_extractor(self):
        cfg = [64, 64, 'M', 128]
        extractor = make_layers(cfg, batch_norm=False)
        ckpt = torch.load(self.cfg.extractor_weights)
        extractor.load_state_dict(ckpt, strict=False)
        for param in extractor.parameters():
            param.requires_grad_(False)
        extractor.to(self.device)
        return extractor

    def tune(self, model, hr_t, lr_t, n_iters=1):
        """
        Tunes model inplace
        """
        with torch.no_grad():
            hr_feat = self.extractor(hr_t)

        model.train()
        for i in range(n_iters):
            sr_t = model(lr_t)
            sr_feat = self.extractor(sr_t)
            loss = self.content_loss(sr_feat, hr_feat)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        model.eval()
