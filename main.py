import argparse
from data import SRDataset, to_array
from config import cfg
from tuners import PerceptualLossTuner, PixelLossTuner, TunerTypes
import torch
from torchsr.models import carn
from torch.utils.data import DataLoader
import cv2
import numpy as np
import gc
import piq
import os
from pathlib import Path
import sys


def parse_args(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument('--tuner', choices=['perceptual', 'pixel'], default='perceptual',
                    help='Tuner type')
    ap.add_argument('--visualize', action='store_true',
                    help='Whether to visualize super resolution result')
    return ap.parse_args(argv)


def main(args: argparse.Namespace):
    # Read params
    scale = cfg.scale
    device = cfg.device
    tuner_num_iters = cfg.tuner_num_iters
    visualize = args.visualize

    # Create model and tuner
    model = carn(scale=scale, pretrained=True)
    if args.tuner == TunerTypes.PerceptualLossTuner.value:
        tuner = PerceptualLossTuner(model, device)
    elif args.tuner == TunerTypes.PixelLossTuner.value:
        tuner = PixelLossTuner(model, device)
    else:
        raise NotImplementedError()

    # Create dataset
    dataset = SRDataset(cfg.hr_folder, cfg.lr_folder, scale=scale)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=1,
        shuffle=False,
    )

    diffs = []
    model.to(device)
    for i, (hr_t, lr_t, name) in enumerate(dataloader):
        # Handle data from loader
        name = name[0]
        lr_t = lr_t.to(device)
        hr_t = hr_t.to(device)

        # Reload untuned weights
        model.load_pretrained()

        with torch.no_grad():
            # Inference untuned model
            sr_t = model(lr_t)
            # Calculate metrics
            psnr_orig = piq.psnr(hr_t.clamp(0, 1), sr_t.clamp(0, 1))

        # Tune the model
        tuner.tune(model, hr_t, lr_t, tuner_num_iters)

        with torch.no_grad():
            # Inference tuned model
            sr_t_tuned = model(lr_t)
            # Calculate metrics
            psnr_tuned = piq.psnr(hr_t.clamp(0, 1).cuda(), sr_t_tuned.clamp(0, 1).cuda())

        print(f'Image {name} | PSNR before {psnr_orig} | PSNR after {psnr_tuned}')
        # Save difference for further analyzation
        diffs.append((psnr_tuned - psnr_orig).cpu().numpy())

        if visualize:
            # Convert tensors to array
            sr = to_array(sr_t)
            sr_mod = to_array(sr_t_tuned)

            # Save super resolution images
            visualization_folder = cfg.vis_folder
            Path(visualization_folder).mkdir(parents=True, exist_ok=True)
            # Save original sr prediction
            cv2.imwrite(os.path.join(visualization_folder, name), sr)

            # Save prediction after tuning
            new_name = name.split('.')
            new_name[-2] = new_name[-2] + '_mod'
            new_name = '.'.join(new_name)
            cv2.imwrite(os.path.join(visualization_folder, new_name), sr_mod)

    diffs = np.array(diffs)
    print(f'Metrics improvement: Mean {np.mean(diffs)} | Median {np.median(diffs)} | Std {np.std(diffs)}')


if __name__ == '__main__':
    main(parse_args(sys.argv[1:]))
