from easydict import EasyDict

cfg_perceptual = EasyDict()
cfg_pixel = EasyDict()

# Learning rate for the tuner optimizer
cfg_perceptual.lr = 6e-4
cfg_perceptual.extractor_weights = 'assets/weights/vgg_layers.pth'

# Learning rate for the tuner optimizer
cfg_pixel.lr = 3e-4  # on higher lr's psnr may degrade
