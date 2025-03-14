import numpy as np
import cv2
import numpy as np
from math import *
import pyclipper
import base64
from PIL import Image, ImageDraw, ImageFont
import sys
sys.path.append('.')
from utils.calsize import calsize
from utils.calsim import new_calsim
from utils.baozha import getContours
from utils.floorplanseg import FloorplanSeg
from utils.floorplandec import main
from utils.kuang_seg import parse_args as kuang_parse_args
from utils.kuang_seg import mainseg as kuang_mainseg
from utils.kuang_seg import get_kuang_wall
from utils.kuang_detection import detect_corners, calw_h
from utils.kuang_detection import calsize as kuangcalsize
#实验部分子空间匹配的结果图展示：子空间二值图-案例-相似度（或者还有各种属性说明）
#举例说明流程
#输入一张户型图即一个框，户型图只提取卧室的爆炸图，案例数据生成
#计算该爆炸图和案例数据的相似度
#画图：画出子空间二值图-案例-相似度（或者还有各种属性说明）的一张图

#获得户型图某个空间的爆炸图数据，即轮廓、尺寸、门窗位置
#初始化案例空间和爆炸图的房间类型列表
kongjian = ["客厅", "餐厅", "客餐厅", "主卧", "主卧带卫生间", "次卧", "独立卫生间", "玄关", "走廊", "阳台", "厨房", "衣帽间", "书房", "厨餐结合", "玄关餐厅结合"]
allroomtype = ["balcony", "diningroom", "livingroom", "bedroom", "masterbedroom", "kictchen", "bathroom", "masterbathroom"]
allroomtype = ["阳台", "餐厅", "客厅", "次卧", "主卧", "厨房", "独立卫生间", "主卫", "玄关"]
#预先定义好户型分割api
floorplanseg_api = FloorplanSeg()
def get_room_baozha(pic_path):
    contents = cv2.imdecode(np.fromfile(pic_path, dtype=np.uint8), -1)
    cv2.imwrite("./segresult/floorplan.jpg", contents)

    #先进行目标检测和语义分割
    # 对原图进行目标检测，剪裁出户型图重点区域
    crop_result = main()

    #如果目标检测出重点户型图区域，那么进行语义分割和爆炸图的图片就是crop.jpg和crop(2).png，否则是户型图原图和floorplan(2).png
    segpic = "./segresult/crop.jpg"
    baozhapic = "./segresult/pseudo_color_prediction/crop(2).png"
    if crop_result == None:
        segpic = "./segresult/floorplan.jpg"
        baozhapic = "./segresult/pseudo_color_prediction/floorplan(2).png"

    #语义分割步骤，如果目标检测有结果，此步骤后会得到crop.png和crop(2).png，否则是floorplan.png和floorplan(2).png，两张图片分别对应后处理前的语义分割结果和后处理后的语义分割结果
    #后处理过程在PaddleSeg/core/postprocess
    floorplanseg_api.predict_floorplan16(segpic)

    
    img = cv2.imread(baozhapic)
    hsvinter = []
    balcony = np.array([[150, 255, 128], [150, 255, 128]])
    hsvinter.append(balcony)

    diningroom = np.array([[90, 255, 128], [90, 255, 128]])
    hsvinter.append(diningroom)

    livingroom = np.array([[0, 0, 128], [0, 0, 128]])
    hsvinter.append(livingroom)

    bedroom = np.array([[0, 255, 64], [0, 255, 64]])
    hsvinter.append(bedroom)

    masterbedroom = np.array([[0, 255, 192], [0, 255, 192]])
    hsvinter.append(masterbedroom)

    kictchen = np.array([[45, 255, 128], [45, 255, 128]])
    hsvinter.append(kictchen)

    bathroom = np.array([[20, 255, 192], [20, 255, 192]])
    hsvinter.append(bathroom)

    porch = np.array([[135, 255, 128], [135, 255, 128]])
    doorway = []
    masterbathroom = np.array([[90, 128, 128], [90, 128, 128]])
    hsvinter.append(masterbathroom)
    allbaozhame = []

    #计算户型图中的尺寸信息
    size = calsize("./segresult/floorplan.jpg")

    # for i in range(len(hsvinter)):
    #     if i >= 0:
    #         baozhaotu = getContours(allroomtype[i], size, img, hsvinter[i][0], hsvinter[i][1], 0, 0, 0)

    #获取次卧的结构数据
    baozhaotu = getContours("次卧", size, img, bedroom[0], bedroom[1])
    return baozhaotu
#获得案例数据
def get_kuang_data(kuang_path):
    contents = cv2.imdecode(np.fromfile(kuang_path, dtype=np.uint8), -1)
    cv2.imwrite("./segresult/kuang.png", contents)
    image_path = "./segresult/kuang.png"
    # 对案例进行语义分割
    args = kuang_parse_args()
    kuang_mainseg(args)
    # 识别案例的尺寸信息，即单位像素代表多少mm，注意这里的calsize用的是./utils/kuang_detection.py中的函数，并不是./utils/calsize.py,后者是用于识别户型图的尺寸信息的，二者流程不一样
    size = kuangcalsize(image_path)
    # 识别出案例中的墙门窗数据，墙数据是每个拐角的坐标点，门数据是每个门的中心点坐标，窗数据是每个窗的两个端点坐标
    wall, door, window = detect_corners(image_path, "meiyou", size)
    # 墙拐点改成分割的方法】
    wall = get_kuang_wall("./segresult/pseudo_color_prediction/kuang.png", size)

    # 识别案例的开间和进深数据
    witdh, height = calw_h(image_path, size, wall, door, window)
    # return {"ID": ID, "wall": json.dumps(wall.tolist()), "door": json.dumps(door.tolist()),
    #         "window": json.dumps(window.tolist()), "size": size, "witdh": witdh, "height": height}
    return np.array([contents, wall, door, window, size, witdh, height])


#计算相似度值
def cal_sim_value(baozhadata, kuangdata):
    floorplan1 = np.array([baozhadata[1], baozhadata[2], baozhadata[3]])
    floorplan2 = np.array([kuangdata[2], kuangdata[3], kuangdata[1]])

    rate = new_calsim("次卧", floorplan1, floorplan2, baozhadata[4], kuangdata[4], baozhadata[5], baozhadata[6],
                      kuangdata[5], kuangdata[6], "00000")
    
    if rate > 1:
        rate = 0.99
    #draw result
    # 载入两张图片
    image1 = Image.fromarray(cv2.cvtColor(baozhadata[0], cv2.COLOR_BGR2RGB))  # 替换为图片1的路径
    image2 = Image.fromarray(cv2.cvtColor(kuangdata[0], cv2.COLOR_BGR2RGB))  # 替换为图片2的路径

    # 确保两张图片大小相同
    image1 = image1.resize(image2.size)

    # 将两张图片水平拼接
    combined_image = Image.new('RGB', (image1.width + image2.width + image1.width, image1.height))
    combined_image.paste(image1, box=[0, 0])
    combined_image.paste(image2, box=[image1.width+20, 0])

    # 在合并后的图片旁边添加文字
    font = ImageFont.truetype('utils/Arial.ttf', 30)  # 指定字体和大小
    draw = ImageDraw.Draw(combined_image)
    draw.text((image1.width + image2.width + 30, int(image1.height/2)), "The similarity between subspace and case is:" + str(rate), font=font)  # 文字位置和内容
    font = ImageFont.truetype('utils/Arial.ttf', 200)  # 指定字体和大小
    draw.text((image1.width + image2.width + 200, int(image1.height / 2)+ 10),str(rate), font=font)  # 文字位置和内容

    # 保存新图片
    combined_image.save('combined_image.jpg')
    return rate




if __name__ =="__main__":
    pic_path = "./data/floorplan/image_2.jpg"
    baozhadata = get_room_baozha(pic_path)
    #print(baozhadata)

    kuang_path = "./data/case/C003.jpg"
    kuangdata = get_kuang_data(kuang_path)
    #print(kuangdata)

    #一个户型图中可能有多个次卧，选择第一个做对比
    simvalue = cal_sim_value(baozhadata[1], kuangdata)
    print(simvalue)
    #画图