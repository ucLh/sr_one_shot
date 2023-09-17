from .config import cfg
from .tuners import PerceptualLossTuner, PixelLossTuner
import torch
from torchsr.models import carn
import numpy as np
from time import time
from tqdm import tqdm
import argparse
import sys


def parse_args(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument('--num_iters', required=False, default=50, type=int,
                    help='Number of iterations to run')
    ap.add_argument('--num_warmup_runs', required=False, default=1, type=int,
                    help='Drop first num_warmup_runs before calculating statistics')
    return ap.parse_args(argv)


def main(args: argparse.Namespace):
    # Read params
    scale = cfg.scale
    device = cfg.device
    tuner_num_iters = cfg.tuner_num_iters
    time_num_iters = args.num_iters
    num_warmup_runs = args.num_warmup_runs

    # Init model and tuners
    model = carn(scale=scale, pretrained=True)
    tuners = [PerceptualLossTuner(device), PixelLossTuner(device)]

    model.to(device)

    for tuner in tuners:
        times_tune = []
        for j in tqdm(range(time_num_iters)):
            # Reload model weights
            model.load_pretrained()

            # Init random tensor
            hr_t = torch.randn((1, 3, 1920, 1080))
            lr_t = torch.randn((1, 3, 1920 // scale, 1080 // scale))

            # Move data to device
            lr_t = lr_t.to(device)
            hr_t = hr_t.to(device)

            # Time tune() method
            t1 = time()
            tuner.tune(model, hr_t, lr_t, tuner_num_iters)
            t2 = time()
            times_tune.append(t2 - t1)

        times_tune = np.array(times_tune[num_warmup_runs:])
        print(f'{tuner} tune() method: Mean {np.mean(times_tune)} | Median {np.median(times_tune)} | Std {np.std(times_tune)}')

    times_infer = []
    for j in tqdm(range(time_num_iters)):
        with torch.no_grad():
            # Init random tensor and move it to device
            lr_t = torch.randn((1, 3, 1920 // scale, 1080 // scale))
            lr_t = lr_t.to(device)

            # Time model inference
            t1 = time()
            sr_t = model(lr_t)
            t2 = time()
            times_infer.append(t2 - t1)
    times_infer = np.array(times_infer[num_warmup_runs:])
    print(f'Perfomance for SR model: Mean {np.mean(times_infer)} | Median {np.median(times_infer)} | Std {np.std(times_infer)}')


if __name__ == '__main__':
    main(parse_args(sys.argv[1:]))
