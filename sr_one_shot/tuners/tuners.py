from abc import ABC
from enum import Enum

import torch
from torchvision.models.vgg import make_layers

from .config import cfg_content, cfg_pixel


class TunerTypes(Enum):
    """
    Enum with all tuner types
    """
    ContentLossTuner = 'content'
    PixelLossTuner = 'pixel'


class AbstractTuner(ABC):
    """
    Abstract Tuner class that sets the tune method interface
    """
    def tune(self, model: torch.nn.Module, hr_t: torch.Tensor, lr_t, n_iters: int):
        pass


class PixelLossTuner(AbstractTuner):
    """
    Pixel loss tuner. Performs gradient steps on set super resolution (SR) model using MSE loss.
    MSE loss is applied to SR network output and a normalized tensor of the original high resolution image.
    This approach of MSE loss usage is usually called `pixel loss`, hence the name.
    """
    def __init__(self, device=None):
        self.cfg = cfg_pixel
        self.pixel_loss = torch.nn.MSELoss(reduction='mean')

    def __repr__(self):
        return 'PixelLossTuner'

    def tune(self, model: torch.nn.Module, hr_t: torch.Tensor, lr_t, n_iters: int = 1):
        """
        Tunes model inplace to perform better on a input image. Method expects both high and low resolution
        versions of the same image.
        :param model: Super Resolution model to tune
        :param hr_t: tensor representing input image of high resolution
        :param lr_t: tensor representing input image of low resolution
        :param n_iters: number of gradient steps to perform
        """
        # Create optimizer every time method is called. Otherwise, one global optimizer will store statistics
        # from all of the previous method calls
        optimizer = torch.optim.Adam(model.parameters(), lr=self.cfg.lr)
        model.train()
        for i in range(n_iters):
            # Run low res tensor through SR model
            sr_t = model(lr_t)
            # Compute pixel loss between SR output and original high resolution image
            loss = self.pixel_loss(sr_t, hr_t)
            # Perform a gradient step
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        model.eval()


class ContentLossTuner(AbstractTuner):
    """
    Content loss tuner. Performs gradient steps on set super resolution (SR) model using MSE loss.
    To compute the loss we take SR network output (sr_t) and a normalized tensor of the original
    high resolution image (hr_t).
    We then infer (sr_t) and (hr_t) through the first few layers of a frozen network pretrained on image net (vgg in
    this case) to obtain tensors (sr_feat) and (hr_feat). MSE loss is then applied to (sr_feat) and (hr_feat), and
    a gradient step is done.
    This approach of MSE loss usage is usually called `content loss`, hence the name.
    I've learnt about this approach from the article
    "Perceptual Losses for Real-Time Style Transfer and Super-Resolution" (https://arxiv.org/pdf/1603.08155.pdf)
    """
    def __init__(self, device):
        self.cfg = cfg_content
        self.content_loss = torch.nn.MSELoss(reduction='mean')
        self.device = device
        self.extractor = self.get_feature_extractor()

    def __repr__(self):
        return 'ContentLossTuner'

    def get_feature_extractor(self) -> torch.nn.Module:
        """
        Load and freeze first few layers of a vgg classifier trained on imagenet
        """
        cfg = [64, 64, 'M', 128]
        extractor = make_layers(cfg, batch_norm=False)
        ckpt = torch.load(self.cfg.extractor_weights)
        extractor.load_state_dict(ckpt, strict=True)
        for param in extractor.parameters():
            param.requires_grad_(False)
        extractor.to(self.device)
        return extractor

    def tune(self, model: torch.nn.Module, hr_t: torch.Tensor, lr_t, n_iters: int = 1):
        """
        Tunes model inplace to perform better on a input image. Method expects both high and low resolution
        versions of the same image.
        :param model: Super Resolution model to tune
        :param hr_t: tensor representing input image of high resolution
        :param lr_t: tensor representing input image of low resolution
        :param n_iters: number of gradient steps to perform
        """
        # Create optimizer every time method is called. Otherwise, one global optimizer will store statistics
        # from all of the previous method calls
        optimizer = torch.optim.Adam(model.parameters(), lr=self.cfg.lr)

        # Run high resolution image through the extractor to obtain the target for loss
        with torch.no_grad():
            hr_feat = self.extractor(hr_t)

        model.train()
        for i in range(n_iters):
            # Run low res tensor through SR model
            sr_t = model(lr_t)
            # Run SR output through the extractor
            sr_feat = self.extractor(sr_t)
            # Compute content loss using the extractor output
            loss = self.content_loss(sr_feat, hr_feat)
            # Perform a gradient step
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        model.eval()
