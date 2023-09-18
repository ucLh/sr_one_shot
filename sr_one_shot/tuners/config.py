from easydict import EasyDict

cfg_content = EasyDict()
cfg_pixel = EasyDict()

# Learning rate for the tuner optimizer
cfg_content.lr = 6e-4
cfg_content.extractor_weights = 'assets/weights/vgg_layers.pth'

# Learning rate for the tuner optimizer
cfg_pixel.lr = 3e-4  # on higher lr's psnr may degrade
