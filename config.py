from easydict import EasyDict
from tuners import TunerTypes


cfg = EasyDict()

cfg.scale = 4
cfg.device = 'cpu'
cfg.tuner_num_iters = 1

cfg.hr_folder = 'assets/img/custom_hr'
cfg.lr_folder = 'assets/img/custom_lr'
cfg.visualize = True
cfg.vis_folder = 'assets/img/vis'

cfg.tuner = TunerTypes.PerceptualLossTuner
# cfg.tuner = TunerTypes.PixelLossTuner

