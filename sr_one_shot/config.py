from easydict import EasyDict

# The paramaters the code was tested with, just gathered in one place
cfg = EasyDict()

# Super resolution network scale. The network will produce an image that is `scale` times bigger than the input image
cfg.scale = 4
# Device to run computations on. Tested on cpu
cfg.device = 'cpu'
# Number of gradient descent steps in tuner
cfg.tuner_num_iters = 1


