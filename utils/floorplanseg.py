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
import PIL.Image
import PIL.ImageDraw
import numpy as np
#from gray2pseudo_color import get_color_map_list
#from postprocess import total_process, combine

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


class FloorplanSeg():
    def __init__(self, Debug=False):
        self.args = parse_args()
        self.env_info = get_sys_env()

        if self.args.device == 'gpu' and self.env_info[
                'Paddle compiled with cuda'] and self.env_info['GPUs used']:
            self.place = 'gpu'
        elif self.args.device == 'xpu' and paddle.is_compiled_with_xpu():
            self.place = 'xpu'
        elif self.args.device == 'npu' and paddle.is_compiled_with_npu():
            self.place = 'npu'
        else:
            self.place = 'cpu'

        paddle.set_device(self.place)

        self.Debug = Debug
        if self.Debug:
            self.appidx = "."
        else:
            self.appidx = ""
       
        self.configpath16 = self.appidx + "./configs/floorplan16.yml"
        self.cfg16 = Config(self.configpath16)
        self.cfg16.check_sync_info()
        self.model16 = self.cfg16.model

        self.transforms16 = Compose(self.cfg16.val_transforms)
        #分割算法的模型文件以及保存结果的文件夹
        self.model_path16 = self.appidx + "./model/floorplanseg/model16.pdparams"
        self.save_dir16 = self.appidx + "./segresult"







    
    def predict_floorplan16(self, segpic):
        image_path = segpic
        image_list, image_dir = get_image_list(image_path)
        logger.info('Number of predict images = {}'.format(len(image_list)))

        test_config = get_test_config(self.cfg16, self.args)
        predict(
            self.model16,
            model_path=self.model_path16,
            transforms=self.transforms16,
            image_list=image_list,
            image_dir=image_dir,
            save_dir=self.save_dir16,
            **test_config)



