from easydict import EasyDict

# The paramaters the code was tested with just gathered in one place
cfg = EasyDict()

# Super resolution network scale. The network will produce an image that is `scale` times bigger than the input image
cfg.scale = 4
# Device to run computations on. Tested on cpu
cfg.device = 'cpu'
# Number of gradient descent steps in tuner
cfg.tuner_num_iters = 1

# Path to a directory with high resolution images. Required
cfg.hr_folder = 'assets/img/custom_hr'
# Directory with low resolution images. If the folder is empty the images will be produced automatically via
# high resolution image interpolation. See more in data.py
cfg.lr_folder = 'assets/img/custom_lr'
# Directory where output images will be saved
cfg.vis_folder = 'assets/img/vis'


