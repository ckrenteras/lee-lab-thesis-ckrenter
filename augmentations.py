import cv2
import numpy as np
from functools import partial
import torchvision.transforms
import elasticdeform

def gaussian_noise(img, std_dev, mean=0):
    noise = np.random.normal(mean * 255, std_dev * 255, img.shape)
    return  np.clip((img.astype(np.float32) + noise), 0, 255).astype(np.uint8)

class GaussianNoiseTransform:
    def __init__(self, std_dev, mean=0):
        self.std_dev = std_dev
        self.mean = mean
    def __call__(self, img):
        return gaussian_noise(img, self.std_dev, self.mean)
    
class ElasticDeform:
    def __init__(self, sigma=3, points=6):
        self.sigma = sigma
        self.points = points

    def __call__(self, image, label):
        image_def, label_def = elasticdeform.deform_random_grid(
            [image, label], self.sigma, self.points,
            order=[1, 0], mode='reflect'
        )
        return image_def.astype(np.uint8), label_def.astype(np.uint8)
    
class ContrastBrightness:
    def __init__(self, alpha=1.5, beta=0):
        self.alpha=alpha
        self.beta=beta
    def __call__(self, image):
        return cv2.convertScaleAbs(image, alpha=self.alpha, beta=self.beta)
    
class Rotation:
    def __init__(self, rotation=cv2.ROTATE_90_CLOCKWISE):
        self.rotation=rotation
    def __call__(self, image, label):
        return cv2.rotate(image, self.rotation), cv2.flip(label, self.flip_code)
    
class ImFlip:
    def __init__(self, flip_code=1):
        self.flip_code=flip_code
    def __call__(self, image, label):
        return cv2.flip(image, self.flip_code), cv2.flip(label, self.flip_code)