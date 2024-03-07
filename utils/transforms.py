import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform
import mlx.core as mx
import numpy as np


def get_transform(label, transform_type, height, width):
    # Define data transformations
    if label == 'train':
        if transform_type == 'all':
            return A.Compose([
                                A.RandomResizedCrop(height=height, width=width, scale=(0.8, 1.0)),
                                A.HorizontalFlip(p=0.25),
                                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.3),
                                A.GaussianBlur(p=0.05),
                                MxNormalize(),
                            ])
        elif transform_type == 'blur':
            return A.Compose([
                                A.GaussianBlur(p=0.05),
                                MxNormalize(),
                            ])
        else:
            return A.Compose([
                                MxNormalize(),
                            ])

    else:
        return A.Compose([
                            MxNormalize(),
                        ])


class MxNormalize(ImageOnlyTransform):
    # Normalize data and output as mx array
    def __init__(self, always_apply=False, p=1.0):
        super(MxNormalize, self).__init__(always_apply, p)

    def apply(self, image, **params):
        # Normalize input and cast to mx array
        return mx.array((image / 255.0).astype(np.float32))
