# One-shot training method for super resolution networks

This code takes [CARN](https://arxiv.org/abs/1803.08664) super resolution network 
from [torchSR](https://github.com/Coloquinte/torchSR) library  and provides some one shot tuning methods 
for improving accuracy on a single image provided it's high resolution version.

### Requirements

For the development python3.8 was used. Required packages are listed in requirements.txt.
You can install them using this command.
```shell script
pip3 install -r /path/to/requirements.txt
```

### Data

The code was tested on a custom dataset of images gathered from the Internet. Image sizes are approximately FullHD.
The original HR images lie in [assets/img/custom_hr](./assets/img/custom_hr). The results are saved in 
[assets/img/vis](./assets/img/vis).

### Run

1. [`demo.py`](./sr_one_shot/demo.py) Performs super resolution network inference and it's tuning, compares psnr and 
saves image results.

Run the demo:
```shell script
python3 -m sr_one_shot.demo --visualize
```
 
The script supports the following CLI:
 - [__`tuner`__](./sr_one_shot/demo.py#L20) - Tuner type. Allowed values: `perceptual`, `pixel`.
 - [__`visualize`__](./sr_one_shot/demo.py#L30) - Whether to save image results.
 - [__`hr_folder`__](./sr_one_shot/demo.py#L22) - Path to the input. Should be a path to a directory with high resolution images.
 - [__`lr_folder`__](./sr_one_shot/demo.py#L24) - Directory with low resolution images. If the folder is empty the images will be
   produced automatically via high resolution image interpolation. Image names must match with the names in --hr_folder,
   otherwise they will be produced automatically. See more in [data.py](./sr_one_shot/data.py).
 - [__`vis_folder`__](./sr_one_shot/demo.py#L28) - Directory where output images will be saved.

2. [`time.py`](./sr_one_shot/time.py) Performs time measurements of SR net
Run the script
```shell script
python3 -m sr_one_shot.time
```

The script supports the following CLI:
 - [__`num_iters`__](./sr_one_shot/time.py#L16) - Number of iterations to run measurements
 - [__`num_warmup_runs`__](./sr_one_shot/time.py#L18) - Drop first num_warmup_runs before calculating statistics.

### Tests

You can run tests using
```shell script
pytest tests/test.py 
```