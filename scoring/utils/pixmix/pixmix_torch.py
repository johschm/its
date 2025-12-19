# Copyright 2019 Google LLC
# Adapted for PyTorch and torchvision functional API
#
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import random
import numpy as np
import torch
from torch import Tensor
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as F

IMAGE_SIZE = 32

#########################################################
#################### AUGMENTATIONS ######################
#########################################################

def int_parameter(level, maxval):
    return int(level * maxval / 10)

def float_parameter(level, maxval):
    return float(level) * maxval / 10.

def sample_level(n):
    return np.random.uniform(0.1, n)

# -------------------
# Functional augmentations
# -------------------
def autocontrast(img, _): return F.autocontrast(img)
def equalize(img, _): return F.equalize(img)

def posterize(img, level):
    level = int_parameter(sample_level(level), 4)
    return F.posterize(img, 4 - level)

def rotate(img, level):
    degrees = int_parameter(sample_level(level), 30)
    if np.random.random() > 0.5: degrees = -degrees
    return F.rotate(img, degrees, interpolation=T.InterpolationMode.BILINEAR)

def solarize(img, level):
    level = int_parameter(sample_level(level), 256)
    return F.solarize(img, 256 - level)

def color(img, level):
    factor = float_parameter(sample_level(level), 1.8) + 0.1
    return F.adjust_saturation(img, factor)

def contrast(img, level):
    factor = float_parameter(sample_level(level), 1.8) + 0.1
    return F.adjust_contrast(img, factor)

def brightness(img, level):
    factor = float_parameter(sample_level(level), 1.8) + 0.1
    return F.adjust_brightness(img, factor)

def sharpness(img, level):
    factor = float_parameter(sample_level(level), 1.8) + 0.1
    return F.adjust_sharpness(img, factor)

# -------------------
# Affine-like transforms_old (functional)
# -------------------
def shear_x(img, level):
    level = float_parameter(sample_level(level), 0.3)
    if np.random.random() > 0.5: level = -level
    return F.affine(img, angle=0, translate=[0,0], scale=1.0,
                    shear=[level*180/np.pi,0], interpolation=T.InterpolationMode.BILINEAR)

def shear_y(img, level):
    level = float_parameter(sample_level(level), 0.3)
    if np.random.random() > 0.5: level = -level
    return F.affine(img, angle=0, translate=[0,0], scale=1.0,
                    shear=[0, level*180/np.pi], interpolation=T.InterpolationMode.BILINEAR)

def translate_x(img, level):
    shift = int_parameter(sample_level(level), IMAGE_SIZE//3)
    if np.random.random() > 0.5: shift = -shift
    return F.affine(img, angle=0, translate=[shift,0], scale=1.0,
                    shear=[0,0], interpolation=T.InterpolationMode.BILINEAR)

def translate_y(img, level):
    shift = int_parameter(sample_level(level), IMAGE_SIZE//3)
    if np.random.random() > 0.5: shift = -shift
    return F.affine(img, angle=0, translate=[0,shift], scale=1.0,
                    shear=[0,0], interpolation=T.InterpolationMode.BILINEAR)

augmentations = [
    autocontrast, equalize, posterize, rotate, solarize,
    #shear_x, shear_y, translate_x, translate_y
]

augmentations_all = augmentations + [color, contrast, brightness, sharpness]

#########################################################
######################## MIXINGS ########################
#########################################################

def get_ab(beta):
    if np.random.random() < 0.5:
        a = np.float32(np.random.beta(beta,1))
        b = np.float32(np.random.beta(1,beta))
    else:
        a = 1 + np.float32(np.random.beta(1,beta))
        b = -np.float32(np.random.beta(1,beta))
    return a,b

def add(img1, img2, beta):
    a,b = get_ab(beta)
    img1, img2 = img1*2-1, img2*2-1
    out = a*img1 + b*img2
    return torch.clamp((out+1)/2,0,1)

def multiply(img1, img2, beta):
    a,b = get_ab(beta)
    img1,img2 = img1*2, img2*2
    out = (img1**a)*(img2.clamp_min(1e-37)**b)
    return torch.clamp(out/2,0,1)

mixings = [add, multiply]

#########################################################
################### PIXMIX CLASS #######################
#########################################################

class PixMixTransform:
    """PyTorch functional PixMix transform, adapted from Google Augmentations."""

    def __init__(self, mixing_set, image_size=32,
                 mean=None,
                 std=None,
                 aug_severity=1, k=2, beta=3.0):
        self.mixing_set = mixing_set
        self.size = image_size
        self.aug_severity = aug_severity
        self.k = k
        self.beta = beta
        self.ops = augmentations_all

        self.tensorize = T.ToTensor()
        self.normalize = T.Normalize(mean,std)
        self.resize = T.Resize((image_size,image_size), interpolation=T.InterpolationMode.BILINEAR)

    def _augment_input(self, img):
        op = random.choice(self.ops)
        return op(img, self.aug_severity)

    def _sample_mixing_image(self):
        idx = random.randint(0,len(self.mixing_set)-1)
        zw = self.mixing_set[idx]
        img = zw[0] if isinstance(zw,tuple) else zw
        if not isinstance(img, Image.Image):
            img = T.ToPILImage()(img)
        return img

    def __call__(self, orig: Image.Image):
        mixing_image = self._sample_mixing_image()
        orig = self.resize(orig)
        if random.random()<0.5:
            mixed = self.tensorize(self._augment_input(orig))
        else:
            mixed = self.tensorize(orig)

        for _ in range(np.random.randint(self.k+1)):
            if random.random()<0.5:
                aug_copy = self.tensorize(self._augment_input(orig))
            else:
                aug_copy = self.tensorize(self.resize(mixing_image))

            mixed_op = random.choice(mixings)
            mixed = mixed_op(mixed, aug_copy, self.beta)
            mixed = torch.clamp(mixed,0,1)

        return self.normalize(mixed)
