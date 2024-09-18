"""
创建人：MECH
创建时间：2022/02/01
功能描述：encoder类，用bert编码文本，融合了whitening方法，可以有效的提高bert的检索能力
    支持使用bert_service，利用远端gpu能力encode bert
功能更新：支持自动whitening，支持指定embedding输出位置和层
"""

import os
import time
from typing import List

from torchvision import transforms


class BertEncoder(BaseEncoder):
    """
    利用bert encode文本数据
    """
    def __init__(
            self,
            model_path: str = '',
    ):
       self.model = torch.load(model_path)
       self.transform = transforms.Compose([
             transforms.Resize((256, 256)),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
             ])

    def img_preprocess(imgs)
        return self.transform(imgs)


    def encode(self, imgs):
       return self.model(img_preprocess(imgs))