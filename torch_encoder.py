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
import torch
import os
import sys
import onnx
import onnxruntime as ort
import time
import torch
from PIL import Image
import numpy as np
from torchvision import transforms
from base_encoder import BaseEncoder


class OnnxEncoder(BaseEncoder):
    """
    利用bert encode文本数据
    """

    def __init__(
            self,
            model_path: str = '/share/home/zhoushenghua/arcface/src/model.onnx',
    ):

        # self.transform = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        # ])
        self.sess = ort.InferenceSession(model_path)
        self.sess.set_providers(['CUDAExecutionProvider'], [ {'device_id': 0}])
        self.input_name = self.sess.get_inputs()[0].name

    def img_preprocess(self, imgs):
        re_imgs = []
        for img in imgs:
            re_imgs.append(self.transform(img))

        return np.array([np.array(img) for img in re_imgs])

    def encode(self, imgs):
        # imgs = self.img_preprocess(imgs)
        pred_onnx = self.sess.run(None, {self.input_name: np.array(imgs)})
        return pred_onnx
