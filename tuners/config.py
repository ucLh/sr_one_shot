from easydict import EasyDict

cfg_perceptual = EasyDict()
cfg_pixel = EasyDict()

cfg_perceptual.lr = 3e-4
cfg_perceptual.extractor_weights = 'assets/weights/vgg_light.pth'

cfg_pixel.lr = 3e-4
