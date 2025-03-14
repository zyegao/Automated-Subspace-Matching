# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os

import paddle

from paddleseg.cvlibs import manager, Config
from paddleseg.utils import get_sys_env, logger, get_image_list
from paddleseg.core import predict
from paddleseg.transforms import Compose
import cv2
import numpy as np
from math import *
from paddleocr import PaddleOCR, draw_ocr
def parse_args():
    parser = argparse.ArgumentParser(description='Model prediction')

    # params of prediction
    parser.add_argument(
        "--config", dest="cfg", help="The config file.", default=None, type=str)
    parser.add_argument(
        '--model_path',
        dest='model_path',
        help='The path of model for prediction',
        type=str,
        default=None)
    parser.add_argument(
        '--image_path',
        dest='image_path',
        help='The image to predict, which can be a path of image, or a file list containing image paths, or a directory including images',
        type=str,
        default=None)
    parser.add_argument(
        '--save_dir',
        dest='save_dir',
        help='The directory for saving the predicted results',
        type=str,
        default='./output/result')

    # augment for prediction
    parser.add_argument(
        '--aug_pred',
        dest='aug_pred',
        help='Whether to use mulit-scales and flip augment for prediction',
        action='store_true')
    parser.add_argument(
        '--scales',
        dest='scales',
        nargs='+',
        help='Scales for augment',
        type=float,
        default=1.0)
    parser.add_argument(
        '--flip_horizontal',
        dest='flip_horizontal',
        help='Whether to use flip horizontally augment',
        action='store_true')
    parser.add_argument(
        '--flip_vertical',
        dest='flip_vertical',
        help='Whether to use flip vertically augment',
        action='store_true')

    # sliding window prediction
    parser.add_argument(
        '--is_slide',
        dest='is_slide',
        help='Whether to prediction by sliding window',
        action='store_true')
    parser.add_argument(
        '--crop_size',
        dest='crop_size',
        nargs=2,
        help='The crop size of sliding window, the first is width and the second is height.',
        type=int,
        default=None)
    parser.add_argument(
        '--stride',
        dest='stride',
        nargs=2,
        help='The stride of sliding window, the first is width and the second is height.',
        type=int,
        default=None)

    # custom color map
    parser.add_argument(
        '--custom_color',
        dest='custom_color',
        nargs='+',
        help='Save images with a custom color map. Default: None, use paddleseg\'s default color map.',
        type=int,
        default=None)

    # set device
    parser.add_argument(
        '--device',
        dest='device',
        help='Device place to be set, which can be GPU, XPU, NPU, CPU',
        default='gpu',
        type=str)

    return parser.parse_args()


def get_test_config(cfg, args):

    test_config = cfg.test_config
    if 'aug_eval' in test_config:
        test_config.pop('aug_eval')
    if args.aug_pred:
        test_config['aug_pred'] = args.aug_pred
        test_config['scales'] = args.scales

    if args.flip_horizontal:
        test_config['flip_horizontal'] = args.flip_horizontal

    if args.flip_vertical:
        test_config['flip_vertical'] = args.flip_vertical

    if args.is_slide:
        test_config['is_slide'] = args.is_slide
        test_config['crop_size'] = args.crop_size
        test_config['stride'] = args.stride

    if args.custom_color:
        test_config['custom_color'] = args.custom_color

    return test_config


def mainseg(args, config_path=None, model_path=None):
    env_info = get_sys_env()

    if args.device == 'gpu' and env_info[
            'Paddle compiled with cuda'] and env_info['GPUs used']:
        place = 'gpu'
    elif args.device == 'xpu' and paddle.is_compiled_with_xpu():
        place = 'xpu'
    elif args.device == 'npu' and paddle.is_compiled_with_npu():
        place = 'npu'
    else:
        place = 'cpu'

    paddle.set_device(place)
    # if not args.cfg:
    #     raise RuntimeError('No configuration file specified.')
    if not isinstance(config_path,type(None)):
        configpath = config_path
    else:
        configpath = "./configs/kuangdetection.yml"
    #configpath = "./configs/kuangdetection.yml"
    cfg = Config(configpath)
    cfg.check_sync_info()

    msg = '\n---------------Config Information---------------\n'
    msg += str(cfg)
    msg += '------------------------------------------------'
    logger.info(msg)

    model = cfg.model
    transforms = Compose(cfg.val_transforms)
    #image_list, image_dir = get_image_list(args.image_path)
    if not isinstance(config_path,type(None)):
        image_path = "../segresult/kuang.png"
    else:
        image_path = "./segresult/kuang.png"
    #image_path = "./test/kuang.jpg"
    image_list, image_dir = get_image_list(image_path)
    logger.info('Number of predict images = {}'.format(len(image_list)))

    test_config = get_test_config(cfg, args)
    if not isinstance(model_path, type(None)):
        model_path = model_path
        save_dir = "../segresult"
    else:
        model_path = "./model/kuangseg/newmodel.pdparams"
        save_dir = "./segresult"

    predict(
        model,
        model_path=model_path,
        transforms=transforms,
        image_list=image_list,
        image_dir=image_dir,
        save_dir=save_dir,
        **test_config)

def get_kuang_wall(image_path, size):
    img = cv2.imread(image_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    room = np.array([[60, 255, 128], [60, 255, 128]])
    low_hsv = room[0]
    high_hsv = room[1]
    mask = cv2.inRange(hsv, lowerb=low_hsv, upperb=high_hsv)
    cannyPic = mask

    # 找轮廓
    contours, hierarchy = cv2.findContours(cannyPic, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    guanceimage = img.copy()
    wall = []
    for c in contours:
        # print(c)
        #cv2.drawContours(img, c, -1, (0, 255, 255), 1)  # 红色
        x, y, w, h = cv2.boundingRect(c)
        if w*size <= 500 or h*size<=500:
            continue
        poly = cv2.approxPolyDP(c, 1, True)
        poly = Rcvnr(poly, size)

        poly = Rcvnr2(poly)

        poly = Rcvnr3(poly)
        poly = Rcvnr(poly, size)

        #print("墙的点数为:", len(poly), poly)
        for data in poly:
            wall.append(data[0])
        guanceimage = cv2.polylines(guanceimage, [poly], True, (0, 0, 0), 10)
        # cv2.namedWindow("room", 0)
        # cv2.imshow("room", guanceimage)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    return np.array(wall)
# Room contour vertex number reduction
# 去除距离近以及共线的点
def Rcvnr(poly,size):
    done = False
    while not done:
        if len(poly) <=4:
            done = True
            break
        for i in range(len(poly)):
            if i >= len(poly) - 1:
                d = sqrt((poly[0][0][0] - poly[i][0][0]) ** 2 + (poly[0][0][1] - poly[i][0][1]) ** 2)
                newdata = [int((poly[0][0][0] + poly[i][0][0]) / 2), int((poly[0][0][1] + poly[i][0][1]) / 2)]
            else:
                d = sqrt((poly[i + 1][0][0] - poly[i][0][0]) ** 2 + (poly[i + 1][0][1] - poly[i][0][1]) ** 2)
                newdata = [int((poly[i][0][0] + poly[i + 1][0][0]) / 2), int((poly[i][0][1] + poly[i + 1][0][1]) / 2)]

            if i == len(poly) - 2:
                v1 = poly[i + 1][0] - poly[i][0]
                v2 = poly[0][0] - poly[i + 1][0]
            elif i == len(poly) - 1:
                v1 = poly[0][0] - poly[i][0]
                v2 = poly[1][0] - poly[0][0]
            else:
                v1 = poly[i + 1][0] - poly[i][0]
                v2 = poly[i + 2][0] - poly[i + 1][0]
            # 计算两向量夹角
            vector_dot_product = np.dot(v1, v2)
            arccos = np.arccos(vector_dot_product / (np.linalg.norm(v1) * np.linalg.norm(v2)))
            angle = np.degrees(arccos)
            # condition a
            
            if d*size <= 200:
                if i == len(poly) - 1:
                    index = [0]
                    poly = np.delete(poly, index, 0)
                else:
                    index = [i + 1]
                    poly = np.delete(poly, index, 0)
                break
                # condition b
            if angle < 10 or angle > 160:
                if i == len(poly) -1:
                    index = [0]
                else:
                    index = [i + 1]
                poly = np.delete(poly, index, 0)
                break

            if i == len(poly) - 1:
                done = True
    return poly

#让所有的线变得水平或者竖直
def Rcvnr2(poly):
    hvec = [1, 0]
    v = [0, 1]
    x, y, w, h = cv2.boundingRect(poly)
    x = int(x + w/2)
    y = int(y + h/2)
    #print(x, y, w, h)
    done = False
    while not done:
        for i in range(len(poly)):
            if i == len(poly) - 1:
                v1 = poly[i][0]
                v2 = poly[0][0]
            else:
                v1 = poly[i][0]
                v2 = poly[i+1][0]
            # 这里不应该用一个范围去删减不平线，只要x或y轴有一个不相等就代表不平,但是要先判断这条线是垂直还是平行的，
            vec = v2 - v1
            vector_dot_product = np.dot(hvec, vec)
            arccos = np.arccos(vector_dot_product / (np.linalg.norm(hvec) * np.linalg.norm(vec)))
            angle = np.degrees(arccos)
            #print(angle, vec, hvec)
            if abs(angle-90) > 45:
                if v1[1] >= y:
                    if abs(v2[1] - v1[1]) > 0:
                        if v2[1] > v1[1]:
                            v1[1] = v2[1]
                        else:
                            v2[1] = v1[1]
                        break
                else:
                    if abs(v2[1] - v1[1]) > 0:
                        if v2[1] < v1[1]:
                            v1[1] = v2[1]
                        else:
                            v2[1] = v1[1]
                        break
            else:
                if v1[0] >= x:
                    if abs(v2[0] - v1[0]) > 0:
                        if v2[0] > v1[0]:
                            v1[0] = v2[0]
                        else:
                            v2[0] = v1[0]
                        break
                else:
                    if abs(v2[0] - v1[0]) > 0:
                        if v2[0] < v1[0]:
                            v1[0] = v2[0]
                        else:
                            v2[0] = v1[0]
                        break
            if i == len(poly) - 1:
                done = True
    return poly

#后处理Rcvnr2的结果，把共线的再删除一遍
def Rcvnr3(poly):
    #print(poly)
    done = False
    while not done:
        if len(poly) <=4:
            done = True
            break
        for i in range(len(poly)):
            if i == len(poly) - 2:
                v1 = poly[i + 1][0] - poly[i][0]
                v2 = poly[0][0] - poly[i + 1][0]
            elif i == len(poly) - 1:
                v1 = poly[0][0] - poly[i][0]
                v2 = poly[1][0] - poly[0][0]
            else:
                v1 = poly[i + 1][0] - poly[i][0]
                v2 = poly[i + 2][0] - poly[i + 1][0]
            # 计算两向量夹角
            vector_dot_product = np.dot(v1, v2)
            arccos = np.arccos(vector_dot_product / (np.linalg.norm(v1) * np.linalg.norm(v2)))
            angle = np.degrees(arccos)

            # condition b
            if angle < 10 or angle > 160:
                if i == len(poly) -1:
                    index = [0]
                else:
                    index = [i + 1]
                poly = np.delete(poly, index, 0)
                break

            if i == len(poly) - 1:
                done = True
    return poly
#识别筐的尺寸信息，即单位像素代表的毫米数
def calsize(image_path):
    ocr = PaddleOCR(use_angle_cls=False, lang="ch")
    result = ocr.ocr(image_path, cls=True)
    result = result[0]
    boxes = []
    txts = []
    scores = []
    num = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    for line in result:
        if line[1][0][0] in num:
            boxes.append(line[0])
            txts.append(int(line[1][0]))
            scores.append(line[1][1])
    #txts[0]为识别出来的文字
    image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8),-1)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 设置黑色的阈值范围
    lower_black = np.array([0, 0, 0], dtype=np.uint8)
    upper_black = np.array([180, 255, 50], dtype=np.uint8)

    # 根据阈值范围创建黑色的掩码
    mask = cv2.inRange(hsv, lower_black, upper_black)

    # 查找轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    wmax = 0
    hmax = 0
    xmax = 0
    ymax = 0
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w >= wmax:
            xmax,ymax,wmax,hmax = x,y,w,h
    #wmax和hmax为黑色区域最大框
    txts = np.array(txts)
    shuzhi = np.max(txts)
    print("检测出来的数字: ", shuzhi)
    size = float(shuzhi)/wmax
    return size

