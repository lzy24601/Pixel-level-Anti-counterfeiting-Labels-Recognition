import copy
import math
import os

import cv2
import numpy as np


# 封装resize函数
def resize_img_keep_ratio(img, target_size):
    # img = cv2.imread(img_name) # 读取图片
    old_size = img.shape[0:2]  # 原始图像大小
    ratio = min(
        float(target_size[i]) / (old_size[i]) for i in range(len(old_size))
    )  # 计算原始图像宽高与目标图像大小的比例，并取其中的较小值
    if ratio > 1:
        ratio = 1
    new_size = tuple(
        [int(i * ratio) for i in old_size]
    )  # 根据上边求得的比例计算在保持比例前提下得到的图像大小
    img = cv2.resize(img, (new_size[1], new_size[0]))  # 根据上边的大小进行放缩
    pad_w = target_size[1] - new_size[1]  # 计算需要填充的像素数目（图像的宽这一维度上）
    pad_h = target_size[0] - new_size[0]  # 计算需要填充的像素数目（图像的高这一维度上）
    top, bottom = pad_h // 2, pad_h - (pad_h // 2)
    left, right = pad_w // 2, pad_w - (pad_w // 2)
    img_new = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, None, (0, 0, 0)
    )
    return img_new


def mkdir(path):  # path是指定文件夹路径
    if os.path.isdir(path):
        print("{} already exists".format(path))
        pass
    else:
        os.makedirs(path)


def roate(src, angle):
    rows, cols = src.shape[:2]
    # 第一个参数旋转中心，第二个参数旋转角度，第三个参数：缩放比例
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)

    # 自适应图片边框大小
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = rows * sin + cols * cos
    new_h = rows * cos + cols * sin
    M[0, 2] += (new_w - cols) * 0.5
    M[1, 2] += (new_h - rows) * 0.5
    w = int(np.round(new_w))
    h = int(np.round(new_h))
    res = cv2.warpAffine(src, M, (w, h))
    return res


def flip(img):
    # 读取图片
    # img = cv2.imread(r'D:\code\python\pytorch-image-models\dataset\false\2\2.jpg')
    src = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 图像翻转
    # 0以X轴为对称轴翻转 >0以Y轴为对称轴翻转 <0X轴Y轴翻转
    img1 = cv2.flip(src, 0)
    img2 = cv2.flip(src, 1)
    img3 = cv2.flip(src, -1)

    # 显示图形
    # titles = ['Source', 'Image1', 'Image2', 'Image3']
    images = [img1, img2, img3]
    return images
    # for i in range(4):
    #     plt.subplot(2,2,i+1), plt.imshow(images[i], 'gray')
    #     plt.title(titles[i])
    #     plt.xticks([]),plt.yticks([])
    #     plt.show()


def illumination(img, strength=None):
    # 读取原始图像
    # img = cv2.imread(r'D:\code\python\pytorch-image-models\dataset\false\2\2.jpg')
    # 获取图像行和列
    rows, cols = img.shape[:2]
    # 设置中心点
    centerX = (rows / 2) + np.random.randint(80)
    centerY = cols / 2 + np.random.randint(100)
    # print (centerX, centerY)
    radius = min(centerX, centerY)
    # print (radius)
    # 设置光照强度
    if strength == None:
        strength = np.random.randint(50, 150)
    # 图像光照特效
    for i in range(rows):
        for j in range(cols):
            # 计算当前点到光照中心距离(平面坐标系中两点之间的距离)
            distance = math.pow((centerY - j), 2) + math.pow((centerX - i), 2)
            # 获取原始图像
            B = img[i, j][0]
            G = img[i, j][1]
            R = img[i, j][2]
            if distance < radius * radius:
                # 按照距离大小计算增强的光照值
                result = (int)(strength * (1.0 - math.sqrt(distance) / radius))
                B = img[i, j][0] + result
                G = img[i, j][1] + result
                R = img[i, j][2] + result
                # 判断边界 防止越界
                B = min(255, max(0, B))
                G = min(255, max(0, G))
                R = min(255, max(0, R))
                img[i, j] = np.uint8((B, G, R))
            else:
                img[i, j] = np.uint8((B, G, R))
    return img


def img_brightness(im, value):
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    if value > 0:
        lim = 255 - value
        v[v > lim] = 255
        v[v <= lim] += value
    else:
        lim = abs(value)
        v[v < lim] = 0
        v[v >= lim] -= lim
    final_hsv = cv2.merge((h, s, v))
    im = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    # cv2.imshow("adjusted", im)
    # cv2.waitKey()
    return im


def img_augmentation(img_root, target_size=[256, 256], aug_nums=20):
    folder = [os.path.join(img_root, name) for name in os.listdir(img_root)]
    for fd in folder:
        img_path = [os.path.join(fd, name) for name in os.listdir(fd)][0]
        img = cv2.imread(img_path)
        tmp1_img = resize_img_keep_ratio(img, target_size)
        cv2.imwrite(os.path.join(fd, os.path.basename(fd) + "_000.jpg"), tmp1_img)
        print(img_path)
        for i in range(1, aug_nums):
            random_roate = np.random.randint(1, 360)
            img2 = roate(copy.deepcopy(img), random_roate)
            img2 = resize_img_keep_ratio(img2, target_size)
            random_factor = np.random.randint(-45, 91)
            img2 = img_brightness(copy.deepcopy(img2), random_factor)
            cv2.imwrite(os.path.join(fd, os.path.basename(fd) + "_%03d.jpg" % i), img2)
        os.remove(img_path)


def imgs_resize(root):
    target = [256, 256]
    for root, dirs, _ in os.walk(root):
        for dir in dirs:
            path = os.path.join(root, dir)
            for r, _, files in os.walk(path):
                for file in files:
                    img_path = os.path.join(r, file)
                    img = cv2.imread(img_path)
                    img = resize_img_keep_ratio(img, target)
                    cv2.imwrite(img_path, img)


# 对从原始图像中分割出的子图做数据增强，按照identity存储
if __name__ == "__main__":
    img_augmentation(img_root=r"D:\code\python\EMHD\Identify\val")
