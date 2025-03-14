import cv2
import numpy as np
from math import *
import pyclipper
import base64
import sys
import json
sys.path.append('..')
from utils.check_master_bedroom import check_master_bedroom
#初始化筐空间和爆炸图的房间类型列表
kongjian = ["客厅", "餐厅", "客餐厅", "主卧", "主卧带卫生间", "次卧", "卫生间", "玄关", "走廊", "阳台", "厨房", "衣帽间", "书房", "厨餐结合", "玄关餐厅结合", "主卧独卫"]

allroomtype = ["balcony", "diningroom", "livingroom", "bedroom", "masterbedroom", "kictchen", "bathroom", "masterbathroom"]
allroomtype = ["阳台", "餐厅", "客厅", "次卧", "主卧", "厨房", "卫生间", "主卧独卫", "玄关"]


#不合法空间的判断，由于识别算法鲁棒性，肯定不能做到100%完全准确识别，尤其是对于不常见的户型图，有可能得到异常的爆炸图，本函数功能是判断爆炸图是否异常，如果异常则选择不返回
'''
异常的条件有：
1. 爆炸图的边数过多,目前暂定大于15为异常
2. 爆炸图为卧室时面积过小
3. 这个不合法只是因为我遇到了一种情况，衣帽间识别成主卧了，且没有窗
4.针对阳台，边数大于等于14为异常
'''
def legal_space(spacewall,spacewindow, size,roomtype):
    #case 1
    if len(spacewall) >= 15 and roomtype != "主卧":
        print(roomtype,"不合法:","边数为: ", len(spacewall))
        return False
    #case 2
    if roomtype == "次卧" or roomtype == "主卧":
        n = len(spacewall)  # 获取顶点数量
        area = 0.0
        for i in range(n):
            x1, y1 = spacewall[i]  # 当前顶点
            x2, y2 = spacewall[(i + 1) % n]  # 下一个顶点，有环绕功能
            area += (x1 * y2) - (x2 * y1)  # 通过公式累加

        space_area = abs(area) / 2.0 * size * size / 1e6
        if space_area <= 2:

            print(roomtype,"不合法:","面积为: ", space_area)
            return False
    #case 3
    if roomtype == "主卧":
        if len(spacewindow) == 0:
            print(roomtype,"不合法:","主卧没有窗")
            return False
        
    #case 4
    if len(spacewall) >= 14 and roomtype == "阳台":
        print(roomtype,"不合法:","边数为: ", len(spacewall))
        return False
    return True  # 计算最终的面积并返回

#爆炸图和筐匹配，输入参数为：当前房间类型， 尺寸， 图像， 该空间hsc下限， 该空间hsv上限， 
def getContours(roomtype, size, img, lowerhsv, highhsv):
    #将图像转换为hsv图像，并通过颜色上下限找出图像中所有该空间的轮廓
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    low_hsv = lowerhsv
    high_hsv = highhsv
    mask = cv2.inRange(hsv, lowerb=low_hsv, upperb=high_hsv)
    # 不做边缘检测，直接检测端点，效果更好
    cannyPic = mask

    # 找轮廓
    contours, hierarchy = cv2.findContours(cannyPic, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    room = []
    roomimage = []
    
    if (roomtype == "餐厅" or roomtype == "客厅" or roomtype == "玄关") and len(contours) >= 1:
        c_0 = 0
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if w * size <= 1000 or h * size <= 1000:
                continue
            if isinstance(c_0,int):
                c_0 = c
            else:
                c_0 = np.row_stack((c_0, c))
        if not isinstance(c_0, int):
            contours = np.array([c_0])
            x, y, w, h = cv2.boundingRect(contours[0])
            #需要逆时针与poly对应，这里爆炸图选择时逆时针排序的，筐的点是顺时针排序的，在相似度对比中对爆炸图的带进行了转换为顺时针
            contours = np.array([[[[x,y]],[[x,y+h]], [[x+w,y+h]], [[x+w, y]]]])


    #以上处理完后，开始对每个轮廓进行处理，得到爆炸图以及墙门窗数据
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        nowroom_witdh = float(w * size)
        nowroom_height = float(h * size)
        print(nowroom_witdh, nowroom_height)
        if roomtype == "次卧" or roomtype == "主卧":
            print("bedroom w h: ", w * size, h * size)
            if w * size <=1580 or h * size <= 1580:
                continue
        elif roomtype == "阳台":
            if w * size <= 1500 and h * size <= 1500:
                continue
        else:
            if w*size<=1010 or h*size<=1010:
                print("空间过小淘汰: ", roomtype)
                continue
        #厨房进行一次处理，因为是矩形的，防止厨房出现曲折爆炸图情况
        if roomtype == "厨房" or roomtype == "独立卫生间":
            c = np.array([[[x, y]], [[x , y+h]], [[x + w, y + h]], [[x + w, y]]])

        #对轮廓进行一次多边形拟合，并进行三次后处理：
        poly = cv2.approxPolyDP(c, 1, True)
        poly = Rcvnr(poly,size)  #去除距离近和共线的点
        poly = Rcvnr2(poly)  #将所有线变得水平或者竖直
        poly = Rcvnr(poly, size)
        poly = Rcvnr2(poly)  # 将所有线变得水平或者竖直
        poly = Rcvnr(poly, size)  #对于上一步的结果再进行一次共线的处理
        poly = Rcvnr2(poly)

        #检测出该空间的墙门窗坐标点，以及对门窗进行一次后处理，删除距离近的点
        wall, windows, door = detect_window_door(poly, hsv, roomtype, size)
        wall, windows, door = wall, delete_min_dis(windows, size), delete_min_dis(door, size)
        print("before delete min ", wall, windows, door)
        finalwall = []
        finaldoor = []
        finalwindow = []

        #poly代表该空间的轮廓，这里是将轮廓向外扩大2个像素，这个步骤仅是为了画出爆炸图，与墙门窗数据与筐的对比无关
        newpoly = equidistant_zoom_contour(poly, 2)
        #防止以上处理有误，进行一次容错处理，处理成矩形
        if len(newpoly) < 4:
            break
        mask = np.zeros(img.shape[:2], np.uint8)

        polyimg1 = cv2.fillPoly(mask, [newpoly], color=(176, 196, 222))
        polyimg1 = cv2.fillPoly(polyimg1, [poly], color=(0, 0, 255))
        #TODO:注意这里，使用的poly，poly就代表墙拐点了
        for i in range(len(poly)):
            finalwall.append(poly[i][0])

        #这里列表door中还是门的两个端点，我们需要处理成门的中心点与筐对应
        for c in range(int(len(door)/2)):
            if len(door) > 2:
                if roomtype == "厨房":
                    doort = 350
                # elif roomtype == "卫生间":
                #     doort = 280
                else:
                    doort = 480
            else:
                doort = 400
            #下面也包括在爆炸图中画出门
            if not cal_two_point_threshold(np.array(door[c * 2]), np.array(door[c * 2 + 1]), size, doort):
                finaldoor.append([int((door[c * 2][0] + door[c * 2 + 1][0])/2), int((door[c * 2][1] + door[c * 2 + 1][1])/2)])
                if door[c * 2][0] == door[c * 2 + 1][0]:
                    if door[c * 2][0] <= x + int(w / 2):
                        cv2.rectangle(polyimg1, [door[c * 2][0] - 2, door[c * 2][1]],
                                      [door[c * 2 + 1][0] + 2, door[c * 2 + 1][1]], (0, 0, 0), -1)
                    else:
                        cv2.rectangle(polyimg1, [door[c * 2][0] - 2, door[c * 2][1]],
                                      [door[c * 2 + 1][0] + 2, door[c * 2 + 1][1]], (0, 0, 0), -1)
                else:
                    if door[c * 2][1] <= y + int(h / 2):
                        cv2.rectangle(polyimg1, [door[c * 2][0], door[c * 2][1] - 2],
                                      [door[c * 2 + 1][0], door[c * 2 + 1][1] + 2], (0, 0, 0), -1)
                    else:
                        cv2.rectangle(polyimg1, [door[c * 2][0], door[c * 2][1] - 2],
                                      [door[c * 2 + 1][0], door[c * 2 + 1][1] + 2], (0, 0, 0), -1)

        #对于窗后处理同上，同样也在爆炸图中画出门
        for c in range(int(len(windows)/2)):
            if roomtype == "主卧" or roomtype == "次卧":
                windowthreshold = 370 
            else:
                windowthreshold = 350
            if not cal_two_point_threshold(np.array(windows[c*2]), np.array(windows[c*2 + 1]), size, windowthreshold):
                finalwindow.append([windows[c*2], windows[c*2 + 1]])
                if windows[c * 2][0] == windows[c * 2 + 1][0]:
                    if windows[c * 2][0] <= x + int(w / 2):
                        cv2.rectangle(polyimg1, [windows[c * 2][0] - 2, windows[c * 2][1]],
                                      [windows[c * 2 + 1][0] - 1, windows[c * 2 + 1][1]], (118, 238, 198), -1)
                    else:
                        cv2.rectangle(polyimg1, [windows[c * 2][0] + 1, windows[c * 2][1]],
                                      [windows[c * 2 + 1][0] + 2, windows[c * 2 + 1][1]], (118, 238, 198), -1)

                else:
                    if windows[c * 2][1] <= y + int(h / 2):
                        cv2.rectangle(polyimg1, [windows[c * 2][0], windows[c * 2][1] - 2],
                                      [windows[c * 2 + 1][0], windows[c * 2 + 1][1] - 1], (118, 238, 198), -1)
                    else:
                        cv2.rectangle(polyimg1, [windows[c * 2][0], windows[c * 2][1] + 1],
                                      [windows[c * 2 + 1][0], windows[c * 2 + 1][1] + 2], (118, 238, 198), -1)
        finalwall,finaldoor,finalwindow = np.array(finalwall), np.array(finaldoor), np.array(finalwindow)
        #这里是处理二连窗和三连窗情况
        if len(finalwindow) >= 2:
            finalwindow = post_windows(finalwindow, size)

            
        # #在这里判断该空间是否合法，合法继续，不合法跳过匹配
        if not legal_space(finalwall,finalwindow, size,roomtype):
            print(roomtype,"不合法")
            continue

        
        #截取爆炸图范围内的图片
        bound = polyimg1.shape
        if y -10 <= 0:
            cropx1 = y
        else:
            cropx1 = y - 10

        if y+h+10 >= bound[0]:
            cropx2 = y+h
        else:
            cropx2 = y+h+10

        if x- 10 <= 0:
            cropy1 = x
        else:
            cropy1 = x- 10

        if x+w+10 >= bound[1]:
            cropy2 = x+w
        else:
            cropy2 = x+w+10
        polyimg1 = polyimg1[cropx1:cropx2, cropy1:cropy2]
        #截取后有可能超出边界

        #*************************#做实验用
        #因为筐的墙坐标是外墙的
        ori_size = size
        #因筐是外墙，但是爆炸图是内墙的，在这里进行一个size的转换，就可以把爆炸图的尺寸进行一个外墙变换
        size = (1 + 400 / nowroom_witdh) * size
        room.append([polyimg1, finaldoor, finalwindow, finalwall, size, nowroom_witdh, nowroom_height])
        #*************************#做实验用

    
    return room

#优化拐角窗户，处理二连窗和三连窗，主要思想是判断是否首尾相连
def post_windows(windows, size):
    done = False
    while not done:
        if len(windows) <= 1:
            done = True
        else:
            for i in range(len(windows)):
                if i == len(windows) - 1:
                    v1 = windows[i][0]
                    v2 = windows[i][1]
                    v3 = windows[0][0]
                    v4 = windows[0][1]
                else:
                    v1 = windows[i][0]
                    v2 = windows[i][1]
                    v3 = windows[i + 1][0]
                    v4 = windows[i + 1][1]
                dis2 = np.linalg.norm(v2 - v3) * size
                if dis2 <= 100:

                    if i == len(windows) - 1:
                        windows[i] = [windows[i][0], windows[0][1]]
                        windows = np.delete(windows, 0, axis=0)
                    else:
                        windows[i] = [windows[i][0], windows[i+1][1]]
                        windows = np.delete(windows, i+1, axis=0)
                    break
                if i == len(windows) - 1:
                    done = True

    return windows


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
            if angle < 10 or angle > 170:
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

#可以缩放多边形
def equidistant_zoom_contour(contour, margin):
    """
    等距离缩放多边形轮廓点
    :param contour: 一个图形的轮廓格式[[[x1, x2]],...],shape是(-1, 1, 2)
    :param margin: 轮廓外扩的像素距离,margin正数是外扩,负数是缩小
    :return: 外扩后的轮廓点
    """
    pco = pyclipper.PyclipperOffset()
    ##### 参数限制，默认成2这里设置大一些，主要是用于多边形的尖角是否用圆角代替
    pco.MiterLimit = 10
    contour = contour[:, 0, :]
    pco.AddPath(contour, pyclipper.JT_MITER, pyclipper.ET_CLOSEDPOLYGON)
    solution = pco.Execute(margin)
    solution = np.array(solution).reshape(-1, 1, 2).astype(int)
    return solution

#判断点坐标是都合法，即是否在图像范围内
def inbound(point, bound):
    if point[1] < bound[0] and point[0] < bound[1] and point[0] >= 0 and point[1] >= 0:
        return True
    else:
        return False
    return False

#判断在图像中该点颜色是否为窗户的
def windowsde(hsv, point, index, xy, nowtype, roomtype):
    bound = hsv.shape
    data = []
    for i in range(index):
        if inbound(point+xy*(i+1), bound):
            newpoint = point+xy*(i+1)
            data.append(hsv[newpoint[1],newpoint[0]])
        if inbound(point-xy*(i+1), bound):
            newpoint = point-xy*(i+1)
            data.append(hsv[newpoint[1], newpoint[0]])

    ishave = False
    wall_window_door = []
    for single_data in data:
        if (single_data == [120, 255, 128]).all():
            wall_window_door.append(0)
        elif (single_data == [30, 255, 128]).all():
            wall_window_door.append(1)
        elif (single_data == [60, 255, 128]).all():
            wall_window_door.append(2)
        elif (single_data == [0, 255, 128]).all():
            #额外加了一种情况，就是识别到了背景
            wall_window_door.append(9)
        else:
            wall_window_door.append(3)
    max_lie = np.argmax(np.bincount(wall_window_door))
    #if wall_window_door.count(0) > 2 and wall_window_door.count(2) == 0:
    # 特殊情况：餐厅，，因为对餐厅的矩形化，户型276728的餐厅矩形化后包含住了一块背景，导致多了一个窗
    if roomtype == "餐厅" and wall_window_door.count(0) > 0 and  wall_window_door.count(9) > 0 and wall_window_door.count(3) == 0:
        return ishave
    if wall_window_door.count(0) > 0 and wall_window_door.count(2) < wall_window_door.count(0):
        if nowtype != "door":
            ishave = True
    return ishave

#判断在图像中该点颜色是否为门的
def doorde(hsv, point, index, xy, nowtype, roomtype):
    bound = hsv.shape
    data = []
    for i in range(index):
        if inbound(point + xy * (i + 1), bound):
            newpoint = point + xy * (i + 1)
            data.append(hsv[newpoint[1], newpoint[0]])
        if inbound(point - xy * (i + 1), bound):
            newpoint = point - xy * (i + 1)
            data.append(hsv[newpoint[1], newpoint[0]])

    ishave = False
    wall_window_door = []
    for single_data in data:
        if (single_data == [30, 255, 128]).all():
            wall_window_door.append(0)
        elif (single_data == [120, 255, 128]).all():
            wall_window_door.append(1)
        elif (single_data == [60, 255, 128]).all():
            wall_window_door.append(2)
        elif (single_data == [0, 255, 128]).all():
            #额外加了一种情况，就是识别到了背景
            wall_window_door.append(9)
        else:
            wall_window_door.append(3)
    max_lie = np.argmax(np.bincount(wall_window_door))

    #如果一个地方只有背景，连墙都没有，说明是卫生间，餐厅等几个空间因为强制矩形化导致的多出了一块
    if wall_window_door.count(9) > 0 and wall_window_door.count(0) == 0 and wall_window_door.count(1) == 0 and wall_window_door.count(2) == 0:
        return ishave
    #if wall_window_door.count(0) > 0 and wall_window_door.count(2) == 0:
    #识别到门的像素点数目大于0，或者（非门窗墙像素点大于零且墙和窗的像素点等于0）
    #优化对于门的判断，如果一个地方同时出现门和墙，则认为该地方门的可能性不大
    # 0: 门   1：窗   2：墙   9：背景    3：房间区域
    if (wall_window_door.count(0) > 0 and wall_window_door.count(2) < wall_window_door.count(0)) or (wall_window_door.count(3) > 0 and wall_window_door.count(2) == 0 and wall_window_door.count(1) == 0 and roomtype == "卫生间"):
        if nowtype != "window":
            ishave = True
    return ishave

#删除距离近的点
def delete_min_dis(points, size):
    if len(points) == 0:
        return points
    done = False
    while not done:
        if len(points) <= 1:
            return []
        for i in range(len(points)):
            if i == len(points) - 1:
                done = True
            if i % 2 != 0:
                continue
            if i >= len(points) - 1:
                d = sqrt((points[0][0] - points[i][0]) ** 2 + (points[0][1] - points[i][1]) ** 2)
            else:
                d = sqrt((points[i + 1][0] - points[i][0]) ** 2 + (points[i + 1][1] - points[i][1]) ** 2)

            if d*size <= 200:
                if i == len(points) - 1:
                    points = np.delete(points, [i], 0)
                    index = [0]
                    points = np.delete(points, index, 0)

                else:
                    index = [i + 1]
                    points = np.delete(points, index, 0)
                    points = np.delete(points, [i], 0)
                break
    return points

#检测出爆炸图中的门和窗的点坐标
#主要思想是逆时针去遍历轮廓上的每个像素点，如果该点是竖直的，就在横向上左右找几个临近点，判断该临近点的颜色值，如果是门的颜色，则该点就是门，通过一些逻辑可以找到所有门和窗的端点
def detect_window_door(poly, hsv, roomtype, size):
    # dot == 0,代表两向量垂直
    h = [1, 0]
    v = [0, 1]
    wall = []
    windows = []
    door = []
    nowvh = "v"
    nowdr = ""
    for i in range(len(poly)):
        if i == len(poly) - 1:
            v1 = poly[i][0]
            v2 = poly[0][0]
        else:
            v1 = poly[i][0]
            v2 = poly[i + 1][0]
        v0 = v2 - v1
        #我希望逆时针去走
        if np.dot(v0, h) == 0:  # 代表当前边与横向边垂直，即当前边是垂直边
            if i == 0:
                nowvh = "v"
            if nowvh == "h":
                nowvh = "v"
                nowdr = "wall"
            if v2[1] > v1[1]:
                step = 1
            else:
                step = -1
            min = v1[1]
            max = v2[1]
            for j in range(min, max, step):
                if doorde(hsv, np.array([v1[0], j]), int(200/size), np.array([1,0]), nowdr, roomtype):
                    if nowdr != "door":
                        if nowdr == "window":
                            windows.append([v1[0], j])
                        door.append([v1[0], j])
                        nowdr = "door"
                elif windowsde(hsv, np.array([v1[0], j]), int(200/size), np.array([1,0]), nowdr, roomtype):
                    if nowdr != "window":
                        if nowdr == "door":
                            door.append([v1[0], j])
                        windows.append([v1[0], j])
                        nowdr = "window"

                else:
                    if nowdr != "wall":
                        if nowdr == "window":
                            windows.append([v1[0], j])
                        elif nowdr == "door":
                            door.append([v1[0], j])

                    nowdr = "wall"
                if j == max-step:
                    if nowdr == "window":
                        windows.append([v1[0], max- step])
                    elif nowdr == "door":
                        door.append([v1[0], max - step])
                    nowdr = "wall"
        else:
            nowdr = ""
            if i == 0:
                nowvh = "h"
            if nowvh == "v":
                nowvh = "h"
                nowdr = "wall"
            if v2[0] > v1[0]:
                step = 1
            else:
                step = -1
            min = v1[0]
            max = v2[0]
            for j in range(min, max, step):
                if doorde(hsv, np.array([j, v1[1]]), int(200/size), np.array([0, 1]), nowdr, roomtype):
                    if nowdr != "door":
                        if nowdr == "window":
                            windows.append([j, v1[1]])
                        door.append([j, v1[1]])
                        nowdr = "door"
                #int(200/size),代表探索一个墙的深度
                elif windowsde(hsv, np.array([j, v1[1]]), int(200/size), np.array([0, 1]), nowdr, roomtype):
                    if nowdr != "window":
                        if nowdr == "door":
                            door.append([j, v1[1]])
                        windows.append([j, v1[1]])
                        nowdr = "window"

                else:
                    if nowdr != "wall":
                        if nowdr == "window":
                            windows.append([j, v1[1]])
                        elif nowdr == "door":
                            door.append([j, v1[1]])

                    nowdr = "wall"
                if j == max-step:
                    if nowdr == "window":
                        windows.append([max-step, v1[1]])
                    elif nowdr == "door":
                        door.append([max -step, v1[1]])
                    nowdr = "wall"
    return wall, windows, door

#计算两点距离是否小于阈值
def cal_two_point_threshold(point1, point2, size, threshold):
    dist = np.linalg.norm(point1-point2)
    print(dist, dist*size)
    if dist*size <= threshold:
        return True
    else:
        return False

