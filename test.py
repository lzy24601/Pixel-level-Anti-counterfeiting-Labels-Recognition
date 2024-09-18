import os
import sys
import time

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn import preprocessing
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder

from backbones import get_model
from faiss_searcher import FaissSearcher
from torch_encoder import OnnxEncoder


# 模型预热
def warm_up(one):
    print("warm_up begin...")
    input = torch.rand(1, 3, 256, 256)
    for i in range(500):
        target = one.encode(input)
    print("warm_up end...")


class ImageFolderWithPath(ImageFolder):
    def __init__(
        self,
        root,
        transform=None,
    ):
        super(ImageFolderWithPath, self).__init__(root, transform)
        self.imgs = self.samples

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        label = os.path.split(path)[1].split("_")[0]
        return sample, path, label
        # return sample, path
        # return sample, label


def get_dataloader(root_dir, batch_size, num_workers=8):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            # 归一化
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    data_sets = ImageFolderWithPath(root_dir, transform)

    data_loader = DataLoader(
        dataset=data_sets,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return data_loader


def get_vec_data(path):
    print("read feature, path_name:{}".format(path))
    t0 = time.perf_counter()
    df = pd.read_csv(path)
    features = []
    labels = []
    for i in range(len(df["feature"])):
        fstr = df["feature"][i][1:-1].split(",")
        labels.append(df["label"][i])
        tmp = []
        for f in fstr:
            tmp.append(float(f))
        features.append(tmp)
    features = np.array(features)
    print("feature shape:{}".format(features.shape))
    print("read feature done total time:{}".format(time.perf_counter() - t0))
    return features, labels


def get_vec_data_with_class(path):
    vec_data_with_class = []
    # for r, _, files in os.walk(path):
    for r, i, files in os.walk(path):
        for file in files:
            vec_path = os.path.join(r, file)
            vec_data, labels = get_vec_data(vec_path)
            # print(file)
            # print(vec_data)
            # exit()

            vec_data_with_class.append(vec_data)
    # for data in vec_data_with_class:
    # print(data)
    return vec_data_with_class


def eval_in_total(
    img_path, vec_path, save_eval_info_path, is_in, save_eval_info_path2="test.csv"
):
    """
    img_path:图片所在的文件路径
    vec_path:向量所在路径
    save_eval_info_path:评测信息保存路径
    is_in:是否在数据库
    """

    inference_time = []
    searcher_time = []
    is_right = []
    img1 = []
    img2 = []
    dist = []
    features1 = []
    features2 = []
    total_imgs = 10000
    # threshold:1.03 acc:0.991
    threshold = 0.68
    # 加载向量查询表
    vec_data_total, labels = get_vec_data(vec_path)

    # index_param = 'HNSW64'
    # 暴力检索
    index_param = "Flat"
    measurement = "l2"
    # 向量查找
    searcher = FaissSearcher(
        vecs=vec_data_total, index_param=index_param, measurement=measurement
    )
    backbone = get_model(
        "r50", with_class=False, dropout=0.0, fp16=True, num_features=256
    ).cuda()
    pytorch_model = torch.load(
        "/share/home/zhoushenghua/arcface/src/work_dirs/myvec_r50_2/model.pt"
    )
    backbone.load_state_dict(pytorch_model)
    backbone.eval()
    t0 = time.perf_counter()
    print("Vector training...")
    searcher.train()
    print("training done total time:{}".format(time.perf_counter() - t0))

    print("start search in total database")

    cnt = 0
    correct_cnt = 0
    data_load = get_dataloader(root_dir=img_path, batch_size=1, num_workers=8)
    for i, data in enumerate(data_load):
        # data是一个list,[img_tensor,类别(如'00000')],imgs torch.Size([1, 3, 256, 256]) <class 'torch.Tensor'>
        imgs, path, label = data
        img1.append(path[0])

        # 反转
        flipped_imgs = torch.flip(imgs, dims=[3])
        t0 = time.perf_counter()
        # feature是一个list,包含一个形状是(1, 256)的ndarray元素
        # feature = one.encode(imgs)
        # pytorch模型
        feature = backbone(imgs.cuda())
        flipped_feature = backbone(flipped_imgs.cuda())
        feature = feature + flipped_feature

        # 归一化
        # feature = preprocessing.normalize(feature[0].cpu().detach().numpy())
        feature = preprocessing.normalize(feature.cpu().detach().numpy())
        inference_time.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        # 返回什么?
        re_dist, index = searcher.search(feature, topK=1)
        img2.append(labels[index[0][0]])

        searcher_time.append(time.perf_counter() - t0)
        if cnt % 100 == 0:
            print("percent: {:.4%}".format(cnt / total_imgs))
        cnt += 1
        diff = np.subtract(feature[0], vec_data_total[index[0][0]])
        features1.append(feature[0])
        features2.append(vec_data_total[index[0][0]])
        cal_dist = np.sum(np.square(diff))
        dist.append("{:.5f}".format(float(cal_dist)))
        # print("cal_dist: {:.4f}, threshold:{}".format(cal_dist, threshold))
        if (cal_dist <= threshold and is_in) or (
            cal_dist > threshold and is_in == False
        ):
            correct_cnt += 1
            is_right.append(True)
        else:
            is_right.append(False)
        # break
    print("acc:{}".format(correct_cnt / cnt))
    dict = {
        "inference_time": inference_time,
        "search_time": searcher_time,
        "is_right": is_right,
        "img1": img1,
        "img2": img2,
        "dist": dist,
    }
    dict2 = {"features1": features1, "features2": features2}

    df = pd.DataFrame(dict)
    df.to_csv(save_eval_info_path)
    df = pd.DataFrame(dict2)
    df.to_csv(save_eval_info_path2)


def get_faiss_with_class(vecs):
    fs = []
    # index_param = 'HNSW64'
    index_param = "Flat"
    measurement = "l2"
    print("get_faiss_with_class() start...")
    for index, vec in enumerate(vecs):
        searcher = FaissSearcher(
            vecs=vec, index_param=index_param, measurement=measurement
        )
        t0 = time.perf_counter()
        print("Vector{} training...".format(index))
        searcher.train()
        print("training done total time:{}".format(time.perf_counter() - t0))
        fs.append(searcher)

    return fs


def get_accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def eval_with_class(
    img_path, vec_path, save_eval_info_path, is_in, save_eval_info_path2
):
    inference_time = []
    searcher_time = []
    is_right = []
    img1 = []
    img2 = []
    dist = []
    features1 = []
    features2 = []
    labels = []
    total_imgs = 10000
    threshold = 0.45
    # 加载向量查询表
    vec_data_with_class = get_vec_data_with_class(vec_path)
    # print(vec_data_with_class)
    # exit()
    # fs是一个列表，包含44个类的查找器searcher
    fs = get_faiss_with_class(vec_data_with_class)

    # one = OnnxEncoder(model_path="tmp.path")
    # warm_up(one)

    backbone = get_model(
        "r50", with_class=True, dropout=0.0, fp16=True, num_features=256
    ).cuda()
    pytorch_model = torch.load(
        "/share/home/zhoushenghua/arcface/src/work_dirs/myvec_r50_withclass_0/model.pt"
    )
    backbone.load_state_dict(pytorch_model)
    backbone.eval()
    cnt = 0
    correct_cnt = 0
    data_load = get_dataloader(root_dir=img_path, batch_size=1, num_workers=8)
    for i, data in enumerate(data_load):
        imgs, path, label = data
        # print(int(label[0]))
        img1.append(path[0])
        # 反转
        flipped_imgs = torch.flip(imgs, dims=[3])
        t0 = time.perf_counter()
        # feature=backbone(imgs.cuda())
        # # 归一化
        # feature = preprocessing.normalize(feature[0].cpu().detach().numpy())
        # feature, _ = one.encode(imgs)
        feature, class_score = backbone(imgs.cuda())

        # exit()
        class_label = int(torch.argmax(class_score))

        flipped_feature, _ = backbone(flipped_imgs.cuda())
        feature = feature + flipped_feature

        feature = preprocessing.normalize(feature.cpu().detach().numpy())
        inference_time.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        re_dist, index = fs[class_label].search(feature, topK=1)
        # img2.append(labels[index[0][0]])

        searcher_time.append(time.perf_counter() - t0)
        if cnt % 100 == 0:
            print("percent: {:.4%}".format(cnt / total_imgs))
        cnt += 1
        # todo
        diff = np.subtract(feature[0], vec_data_with_class[class_label][index[0][0]])
        features1.append(feature[0])
        features2.append(vec_data_with_class[class_label][index[0][0]])
        cal_dist = np.sum(np.square(diff))
        dist.append("{:.5f}".format(float(cal_dist)))
        # print("res_dist:{} cal_dist:{}".format(re_dist, cal_dist))
        if (cal_dist <= threshold and is_in) or (
            cal_dist > threshold and is_in == False
        ):
            correct_cnt += 1
            is_right.append(True)
        else:
            is_right.append(False)
        labels.append(label)
    print("acc:{}".format(correct_cnt / cnt))
    dict = {
        "inference_time": inference_time,
        "search_time": searcher_time,
        "is_right": is_right,
        "img1": img1,
        "dist": dist,
        "label": labels,
    }
    # dict2 = {'features1':features1,'features2':features2}
    df = pd.DataFrame(dict)
    df.to_csv(save_eval_info_path)
    # df = pd.DataFrame(dict2)
    # df.to_csv(save_eval_info_path2)


def main():
    img_path = r"/share/home/zhoushenghua/arcface/search_data/data_for_search_ori/in"

    #    eval_in_total(img_path="/share/home/zhoushenghua/arcface/small_data/data_for_search/in", vec_path="../data/vec/small_data.csv", save_eval_info_path="flat_all.csv", is_in=True,save_eval_info_path2='flat_feature.csv')
    eval_with_class(
        img_path="/share/home/zhoushenghua/arcface/search_data/data_for_search_ori/in",
        vec_path="../data/vec/all_44",
        save_eval_info_path="/share/home/zhoushenghua/arcface/results/all/in_with_class_045.csv",
        is_in=True,
        save_eval_info_path2="/share/home/zhoushenghua/arcface/results/all/in_with_class_Flat_feature.csv",
    )

    # get_vec_data_with_class(r'/share/home/zhoushenghua/arcface/data/vec/vec_44')


if __name__ == "__main__":
    main()
