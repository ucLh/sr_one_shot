import pytest
import sys
import os
sys.path.append(os.path.realpath(os.path.dirname(__file__)+"/.."))
from main import demo
import argparse
import torch
from torchsr.models import carn
from torchvision.transforms.functional import to_tensor
import cv2
from piq import psnr
from tuners import PixelLossTuner, PerceptualLossTuner
import numpy as np


def test_demo_correct():
    namespace = argparse.Namespace(tuner='pixel', visualize=True)
    namespace2 = argparse.Namespace(tuner='perceptual', visualize=True)
    namespace3 = argparse.Namespace(tuner='pixel', visualize=False)

    # Test scenarios run with correct params
    demo(namespace)
    demo(namespace2)
    demo(namespace3)


def test_demo_incorrect():
    namespace_error = argparse.Namespace(tuner='pixel111', visualize=True)

    with pytest.raises(NotImplementedError):
        demo(namespace_error)


def test_consistency():
    hr_img = cv2.imread('assets/img/custom_hr/train.jpg')
    hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2RGB)
    lr_img = cv2.resize(hr_img, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_CUBIC)

    hr_t = to_tensor(hr_img).unsqueeze(0)
    lr_t = to_tensor(lr_img).unsqueeze(0)

    model = carn(4, True)
    model.eval()

    def temp(tuner):
        orig, tuned = [], []
        for i in range(2):
            model.load_pretrained()
            with torch.no_grad():
                sr_t = model(lr_t)
                psnr_orig = psnr(sr_t.clamp(0, 1), hr_t.clamp(0, 1)).cpu().numpy()
                orig.append(psnr_orig)

            tuner.tune(model, hr_t, lr_t, 1)
            with torch.no_grad():
                sr_t = model(lr_t)
                psnr_tuned = psnr(sr_t.clamp(0, 1), hr_t.clamp(0, 1)).cpu().numpy()
                tuned.append(psnr_tuned)

        assert np.allclose(orig[0], orig[1])
        assert np.allclose(tuned[0], tuned[1])

    tuners = [PixelLossTuner(model, 'cpu'), PerceptualLossTuner(model, 'cpu')]
    for tuner in tuners:
        temp(tuner)

