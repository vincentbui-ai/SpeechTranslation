# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from .image_classification import (ImageClassificationConfig,
                                   ImageClassificationTask)
from .image_pretraining import ImagePretrainingConfig, ImagePretrainingTask
from .mae_image_pretraining import (MaeImagePretrainingConfig,
                                    MaeImagePretrainingTask)

__all__ = [
    "ImageClassificationTask",
    "ImageClassificationConfig",
    "ImagePretrainingTask",
    "ImagePretrainingConfig",
    "MaeImagePretrainingTask",
    "MaeImagePretrainingConfig",
]