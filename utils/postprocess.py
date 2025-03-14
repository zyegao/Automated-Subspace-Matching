import cv2
import numpy as np
from scipy import ndimage
import paddle
from paddleseg import utils
from paddleseg.core import infer
from paddleseg.utils import logger, progbar, visualize
import PIL.Image
import PIL.ImageDraw

from utils.gray2pseudo_color import get_color_map_list
from collections import Counter
def fast_hist(im, gt, n=16):
    """
    n is num_of_classes
    """
    k = (gt >= 0) & (gt < n)
    return np.bincount(n * gt[k].astype(int) + im[k], minlength=n ** 2).reshape(n, n)


def flood_fill(test_array, h_max=255):
    """
	fill in the hole
	"""
    input_array = np.copy(test_array)
    el = ndimage.generate_binary_structure(2, 2).astype(int)
    inside_mask = ndimage.binary_erosion(~np.isnan(input_array), structure=el)
    output_array = np.copy(input_array)
    output_array[inside_mask] = h_max
    output_old_array = np.copy(input_array)
    output_old_array.fill(0)
    el = ndimage.generate_binary_structure(2, 1).astype(int)
    while not np.array_equal(output_old_array, output_array):
        output_old_array = np.copy(output_array)
        output_array = np.maximum(input_array, ndimage.grey_erosion(output_array, size=(3, 3), footprint=el))
    return output_array


def fill_break_line(cw_mask):
    broken_line_h = np.array([[0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0],
                              [1, 0, 0, 0, 1],
                              [0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0]], dtype=np.uint8)
    broken_line_h2 = np.array([[0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0],
                               [1, 1, 0, 1, 1],
                               [0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0]], dtype=np.uint8)
    broken_line_v = np.transpose(broken_line_h)
    broken_line_v2 = np.transpose(broken_line_h2)
    cw_mask = cv2.morphologyEx(cw_mask, cv2.MORPH_CLOSE, broken_line_h)
    cw_mask = cv2.morphologyEx(cw_mask, cv2.MORPH_CLOSE, broken_line_v)
    cw_mask = cv2.morphologyEx(cw_mask, cv2.MORPH_CLOSE, broken_line_h2)
    cw_mask = cv2.morphologyEx(cw_mask, cv2.MORPH_CLOSE, broken_line_v2)

    return cw_mask


def fill_break_line2(cw_mask):
    broken_line_h = np.array([[0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0],
                              [1, 0, 0, 0, 0]], dtype=np.uint8)
    broken_line_h2 = np.array([[0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0],
                               [1, 1, 1, 0, 0],
                               [1, 1, 1, 0, 0],
                               [1, 1, 1, 0, 0]], dtype=np.uint8)
    broken_line_v = np.transpose(broken_line_h)
    broken_line_v2 = np.transpose(broken_line_h2)
    cw_mask = cv2.morphologyEx(cw_mask, cv2.MORPH_OPEN, broken_line_h)
    cw_mask = cv2.morphologyEx(cw_mask, cv2.MORPH_OPEN, broken_line_v)
    cw_mask = cv2.morphologyEx(cw_mask, cv2.MORPH_OPEN, broken_line_h2)
    cw_mask = cv2.morphologyEx(cw_mask, cv2.MORPH_OPEN, broken_line_v2)

    return cw_mask


def refine_room_region(cw_mask, rm_ind):
    label_rm, num_label = ndimage.label((1 - cw_mask))
    new_rm_ind = np.zeros(rm_ind.shape)
    for j in range(1, num_label + 1):
        mask = (label_rm == j).astype(np.uint8)
        ys, xs = np.where(mask != 0)
        area = (np.amax(xs) - np.amin(xs)) * (np.amax(ys) - np.amin(ys))
        if area < 100:
            continue
        else:
            room_types, type_counts = np.unique(mask * rm_ind, return_counts=True)
            #old_room_types, old_type_counts = room_types, type_counts
            if len(room_types) > 1:
                room_types = room_types[1:]  # ignore background type which is zero
                type_counts = type_counts[1:]  # ignore background count
            if np.max(type_counts) <= 100:  # 如果连通域中最大的颜色数目过小，可以认为这里分类不好，全置为0
                new_rm_ind[mask == 1] = 0
            elif room_types[np.argmax(type_counts)] == 0:
                new_rm_ind[mask == 1] = 0
            else:
                if room_types[np.argmax(type_counts)] == 6 or room_types[np.argmax(type_counts)] == 12 or room_types[np.argmax(type_counts)] == 5:
                    #在这里单独处理阳台，有一种情况空间一部分是otherroom，一部分是阳台，这个大概率是阳台，这种情况全部弄成阳台

                    if len(room_types) > 1:
                        if room_types[np.argmax(type_counts)] == 12 and room_types[np.argsort(type_counts)[-2]] == 4 and type_counts[np.argsort(type_counts)[-2]] > np.sum(type_counts)/4:
                            new_rm_ind += mask * room_types[np.argsort(type_counts)[-2]] 
                        else:
                            new_rm_ind += mask * rm_ind
                    else:
                        new_rm_ind += mask * rm_ind
                    
                
                else:
                    #这里单独处理一种情况，如果一个空间最大元素是卫生间，第二大元素是主卫，则将其处理成主卫
                    if len(room_types) > 1:
                        if room_types[np.argmax(type_counts)] == 10 and room_types[np.argsort(type_counts)[-2]] == 11 and type_counts[np.argsort(type_counts)[-2]] > np.sum(type_counts)/4:
                            new_rm_ind += mask * 11
                        else:
                            new_rm_ind += mask * room_types[np.argmax(type_counts)]
                    else:
                        # #还有一种特殊情况，户型287212，阳台空间内有背景，然后因为忽略掉了背景，导致阳台空间曲折，下面加回背景
                        ##没有办法在这解决
                        # if room_types[np.argmax(type_counts)] == 4:
                        #     if len(old_room_types) > 1:
                        #         if old_room_types[np.argsort(old_type_counts)[-2]] == 0 and old_type_counts[np.argsort(old_type_counts)[-2]] < np.sum(old_type_counts)/2:
                        #             old_mask = (label_rm == 0).astype(np.uint8)
                        new_rm_ind += mask * room_types[np.argmax(type_counts)]

    return new_rm_ind


def total_process(ori_pred):
    # pred2 = paddle.squeeze(ori_pred)
    # pred2 = pred2.numpy().astype('uint8')
    pred2 = ori_pred
    rm_ind = pred2.copy()
    rm_ind[pred2 == 0] = 0
    rm_ind[pred2 == 1] = 0
    rm_ind[pred2 == 2] = 0
    rm_ind[pred2 == 3] = 0
    # cv2.imwrite('gray_image1.jpg', rm_ind)
    bd_ind = np.zeros(pred2.shape, dtype=np.uint8)
    bd_ind[pred2 == 1] = 1
    bd_ind[pred2 == 2] = 2
    bd_ind[pred2 == 3] = 3
    # cv2.imwrite('gray_image2.jpg', bd_ind)
    wall_c = (bd_ind == 1).astype(np.uint8)
    door_c = (bd_ind == 2).astype(np.uint8)
    window_c = (bd_ind == 3).astype(np.uint8)
    cw_c = (bd_ind > 0).astype(np.uint8)
    # region from room prediction it self
    rm_mask = np.zeros(rm_ind.shape)
    rm_mask[rm_ind > 0] = 1
    # region from close_wall line
    wall_mask = wall_c
    door_mask = door_c
    window_mask = window_c
    cw_mask = cw_c
    # refine close wall mask by filling the grap between bright line
    # change by zyegao 202410223,,继续处理墙门窗了
    # wall_mask = fill_break_line(wall_mask)
    # door_mask = fill_break_line(door_mask)
    # window_mask = fill_break_line(window_mask)
    # cw_mask = fill_break_line(cw_mask)

    new_rm_ind = rm_ind.copy()
    new_rm_ind[cw_mask == 1] = 1
    new_rm_ind[wall_mask == 1] = 1
    new_rm_ind[door_mask == 1] = 2
    new_rm_ind[window_mask == 1] = 3

    fuse_mask = cw_mask + rm_mask
    fuse_mask[fuse_mask >= 1] = 255

    # refine fuse mask by filling the hole
    #加上这个会卡主，暂时删掉
    # fuse_mask = flood_fill(fuse_mask)

    fuse_mask = fuse_mask // 255

    # one room one label
    new_rm_ind = refine_room_region(cw_mask, rm_ind)
    # ignore the background mislabeling
    new_rm_ind = fuse_mask * new_rm_ind
    # new_rm_ind = fuse_mask * rm_ind
    new_rm_ind = new_rm_ind + cw_mask
    new_rm_ind[cw_mask == 1] = 1
    new_rm_ind[wall_mask == 1] = 1
    new_rm_ind[door_mask == 1] = 2
    new_rm_ind[window_mask == 1] = 3

    # new_rm_ind = fill_break_line_one_by_one(new_rm_ind)

    # cv2.imwrite('gray_image3.jpg', new_rm_ind)

    return new_rm_ind


def fill_break_line_one_by_one(new_floorplan):
    # 不处理背景，墙，窗，门
    tmp_floorplan = new_floorplan.copy()
    for i in range(12):
        if np.sum(new_floorplan == i + 4) > 100:
            step_mask = np.zeros(new_floorplan.shape)
            step_mask[new_floorplan <= 3] = 1
            # step_mask[new_floorplan == 0] = 0
            step_mask[new_floorplan == (i + 4)] = 1
            # step_mask = ndimage.binary_fill_holes(step_mask).astype(int)
            step_mask[step_mask >= 1] = 255
            step_mask = flood_fill(step_mask)
            step_mask = step_mask // 255
            tmp_floorplan[step_mask == 1] = i + 4
            tmp_floorplan[new_floorplan == 0] = 0
            tmp_floorplan[new_floorplan == 1] = 1
            tmp_floorplan[new_floorplan == 2] = 2
            tmp_floorplan[new_floorplan == 3] = 3

    return tmp_floorplan

