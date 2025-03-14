#!/usr/bin/python
# -*- coding: UTF-8 -*-
import cv2
import numpy as np
from math import *
from scipy.spatial import ConvexHull
from paddleocr import PaddleOCR, draw_ocr
from matplotlib import pyplot as plt
from shapely.geometry import Polygon
from PIL import Image
#找出框中的墙门窗数据坐标点
def detect_corners(image_path, filename, size):
    # 加载图像
    image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8),-1)
    # 将图像转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 进行角点检测，角点代表墙的拐点，但不是所有的都是，我们需要判断哪些是最外部的那些角点，且那些附近有门窗的点也不是拐点
    corners = cv2.goodFeaturesToTrack(gray, 100, 0.01, 10)  # 参数可根据需要进行调整
    # 将浮点型坐标转换为整数型
    corners = np.int0(corners)
    # 绘制角点并获取坐标
    wall = []
    door = []
    window = []
    #对每一个角点进行处理，找到附近只有墙的点
    for corner in corners:
        x, y = corner.ravel()  # 获取角点坐标
        #识别该点附近有什么像素
        hwall,hdoor,hwindow,hnum = circlehascolor(image,x,y,10)
        #如果该点附近只有墙，则将其加入墙数组
        if hwall==1 and hdoor == 0 and hwindow == 0 and hnum ==0:
            wall.append((x, y))
        else:
            #以下流程得到的门窗没有用，后面用的是其他方法找到门窗
            if hwall==1 and hdoor == 1:
                door.append((x, y))
            elif hwall==1 and hwindow == 1:
                window.append((x, y))
            else:
                continue

    #对上面的墙数组进行处理，找到最外层的一圈点
    polygon = find_wall(image_path, wall, 1)

    #找到窗端点，并将二连窗和三连窗代表的墙点加入到墙数组中
    finlwindow, addpolygon = findwindow(image_path, size)
    for data in addpolygon:
        polygon.append(list(data))
    finlwindow = np.array(finlwindow)

    #对墙的点进行顺时针排序，方便筐对比
    polygon = adjust_pts_order(polygon)
    #对墙点再次处理，去掉共线的中间点
    polygon = Rcvnr3(polygon)
    #去掉墙点中距离非常近的点
    finalwall = np.array(polygon)
    finalwall = del_close_point("wall",finalwall,size)
    # 画出删减后的墙点
    for data in finalwall:
        x, y = int(data[0]), int(data[1])  # 获取角点坐标
        cv2.circle(image, (x, y), 5, (0, 255, 0), -1)  # 绘制红色圆圈表示角点
    #找到所有门的中心点，并去掉距离非常近的
    finldoor = np.array(finddoor(image_path, size))
    finldoor = del_close_point("door", finldoor, size)
    #画出门窗点
    for data in finldoor:
        x, y = int(data[0]),int(data[1])
        cv2.circle(image, (x, y), 5, (255, 255, 0), -1)

    for data in finlwindow:
        x, y = int(data[0][0]),int(data[0][1])
        cv2.circle(image, (x, y), 5, (111, 1, 200), -1)
        x, y = int(data[1][0]), int(data[1][1])
        cv2.circle(image, (x, y), 5, (111, 1, 200), -1)

    # 显示原始图像和绘制角点后的图像
    #print("wall: ",len(finalwall),"door: ",len(finldoor),"window:",len(finlwindow))

    #如果需要看一下结果，可将下面四行注释去掉
    # cv2.namedWindow("Original Image", cv2.WINDOW_NORMAL)
    # cv2.imshow('Original Image', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return finalwall,finldoor,finlwindow

#将坐标点顺时针排序
def adjust_pts_order(pts_2ds):
    #print(pts_2ds)
    ''' sort rectangle points by counterclockwise '''
    cen_x, cen_y = np.mean(pts_2ds, axis=0)
    d2s = []
    for i in range(len(pts_2ds)):

        o_x = pts_2ds[i][0] - cen_x
        o_y = pts_2ds[i][1] - cen_y
        atan2 = np.arctan2(o_y, o_x)
        if atan2 < 0:
            atan2 += np.pi * 2
        d2s.append([pts_2ds[i], atan2])
    d2s = sorted(d2s, key=lambda x:x[1])
    order_2ds = np.array([x[0] for x in d2s])
    return order_2ds

#把共线的再删除一遍
def Rcvnr3(poly):
    done = False
    while not done:
        if len(poly) <=4:
            done = True
            break
        for i in range(len(poly)):
            if i == len(poly) - 2:
                v1 = poly[i + 1] - poly[i]
                v2 = poly[0] - poly[i + 1]
            elif i == len(poly) - 1:
                v1 = poly[0] - poly[i]
                v2 = poly[1] - poly[0]
            else:
                v1 = poly[i + 1] - poly[i]
                v2 = poly[i + 2] - poly[i + 1]
            # 计算两向量夹角
            vector_dot_product = np.dot(v1, v2)
            arccos = np.arccos(vector_dot_product / (np.linalg.norm(v1) * np.linalg.norm(v2)))
            angle = np.degrees(arccos)
            #如果角度小于10度，则删除中间点
            if angle < 10:
                if i == len(poly) -1:
                    index = [0]
                else:
                    index = [i + 1]
                poly = np.delete(poly, index, 0)
                break
            if i == len(poly) - 1:
                done = True
    return poly
#判断点坐标是否合法，在图像内
def inbound(point, bound):
    if point[1] < bound[0] and point[0] < bound[1] and point[0] >= 0 and point[1] >= 0:
        return True
    else:
        return False
    return False

#判断点附近像素类别，是白色背景还是门或者窗
def white_or_other(image, point):  #在像素点上操作会有各种问题，所以改成在一个小矩形内操作
    zuoshang = [point[0],point[1]]
    white = 1  #1代表都是白色，2代表遇到红色，3代表遇到门窗
    bound = image.shape
    for i in range(1):
        for j in range(1):
            new_point = [zuoshang[0] + i, zuoshang[1] + j]
            if inbound(new_point, bound):
                if (image[new_point[1], new_point[0]] == [255, 255, 255]).all() or (image[new_point[1], new_point[0]] == [0, 0, 0]).all():
                    continue
                elif image[new_point[1], new_point[0]][0] >= 200 and image[new_point[1], new_point[0]][1]>= 200 and image[new_point[1], new_point[0]][2]<= 100: #判断有浅蓝色即窗
                    white = 3
                elif image[new_point[1], new_point[0]][0] >= 200 and image[new_point[1], new_point[0]][1]<= 100: #判断有蓝色即门
                    white = 3
                else:
                    if white != 3:
                            white = 2
    return white

#找出最外层的一圈墙点，主要思想是最外层的一圈墙点在上下左右以及上下45度方向等8个方向上一定有一个方向上全都是白色背景
def find_wall(image_path, wall, size):
    image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
    bound = image.shape
    wall_point = []
    #八个方向
    add_point = np.array([[-1,0], [-1,-1], [0,-1], [1,-1], [1,0], [1,1], [0,1], [-1,1]])
    for i in range(len(wall)):
        score_eight_direction = [0, 0, 0, 0, 0, 0, 0, 0]
        # 1代表第一次遇到其他颜色， 2代表第一次遇到白色或黑色， 3代表遇到白色或者黑色后又遇到其他颜色,或者遇到门窗，一个方向上遇到门窗，即使接下来一直白，也说明这个方向不对，参考01.厨房（120个筐）/E006-L型厨房.jpg
        direction_done = [0,0,0,0,0,0,0,0]
        lastcount = 1
        count = 1
        done = False
        while not done:
            for j in range(8):
                if direction_done[j] == 1:
                    continue
                point = np.array(wall[i]) + add_point[j]*count
                if inbound(point, bound):
                    if white_or_other(image, point) == 3:
                        score_eight_direction[j] = 3
                        direction_done[j] += 1
                    elif white_or_other(image, point) == 1:
                        if count == 1:
                            lastcount = count
                            score_eight_direction[j] = 2
                        else:
                            if score_eight_direction[j] == 1:
                                lastcount = count
                                score_eight_direction[j] = 2
                    else:
                        if count == 1:
                            score_eight_direction[j] = 1
                        else:
                            if score_eight_direction[j] == 2:
                                if count - lastcount >= 3:
                                    score_eight_direction[j] = 3
                                    direction_done[j] += 1
                else:
                    direction_done[j] += 1
            count += 1
            if np.array(direction_done).sum() == 8:
                done = True
        for o in range(len(score_eight_direction)):
            if score_eight_direction[o] == 2:
                wall_point.append(wall[i])
                break
    return wall_point

#判断图片中以x，y为中心，r为半径的圆内的像素颜色，有什么颜色的点就会把该颜色对应的元素置为1，比如如果有红色就会把wall置为1
def circlehascolor(image,x,y,r):
    center = [0,0]
    center[0] = x
    center[1] = y
    radius = r
    wall = 0
    door = 0
    window = 0
    num = 0
    bound = image.shape
    #因为有的图片中黑色的尺寸太接近图片边缘导致数组出界,但是不一定只有黑色在边缘，红色也有可能在边缘
    for i in range(center[0]-radius,center[0]+radius):
        for j in range(center[1]-radius,center[1]+radius):
            if not inbound([i,j], bound):
                continue
            if image[j,i][0] >= 200 and image[j,i][1]>= 200 and image[j,i][2]<= 100: #判断有浅蓝色即窗
                window = 1
            if image[j,i][0] >= 200 and image[j,i][1]<= 100: #判断有蓝色及门
                door = 1
            if image[j,i][0] <= 100 and image[j,i][1]<= 100 and image[j,i][2]<= 100: #判断有黑色
                num = 1
                break
            if image[j,i][0] <= 100 and image[j,i][1]<= 100 and image[j,i][2]>= 200: #判断有红色
                wall = 1
        if num == 1:  #因为有的黑色尺寸在边界处，容易导致数组超过图片大小
            break
    return wall,door,window,num

#识别筐的尺寸信息，即单位像素代表的毫米数
def calsize(image_path):
    image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8),-1)
    ocr = PaddleOCR(use_angle_cls=False, lang="ch")
    result = ocr.ocr(image, cls=False)

    result = result[0]
    boxes = []
    txts = []
    scores = []
    num = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    for line in result:
        try:
            boxes.append(line[0])
            txts.append(int(line[1][0]))
            scores.append(line[1][1])
        except:
            print("error size: ", line[1][0])
            continue
        # if line[1][0][0] in num:
        #     boxes.append(line[0])
        #     txts.append(int(line[1][0]))
        #     scores.append(line[1][1])
    #txts[0]为识别出来的文字

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

    biaochi_length_w = 0
    biaochi_length_h = 0
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w >= wmax:
            xmax,ymax,wmax,hmax = x,y,w,h
    #wmax和hmax为黑色区域最大框
    biaochi_length_w = wmax

    txts = np.array(txts)
    if len(txts) > 0:
        shuzhi = np.max(txts)
    else:
        shuzhi = 0

    if shuzhi < 500:
        image_90 = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        result = ocr.ocr(image_90, cls=False)
        result = result[0]
        boxes = []
        txts = []
        scores = []
        for line in result:
            if line[1][0][0] in num:
                boxes.append(line[0])
                txts.append(int(line[1][0]))
                scores.append(line[1][1])
        # txts[0]为识别出来的文字
        txts = np.array(txts)
        if len(txts) > 0:
            shuzhi = np.max(txts)
        else:
            shuzhi = 0
        if shuzhi < 500:
            image_270 = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            result = ocr.ocr(image_270, cls=False)
            result = result[0]
            boxes = []
            txts = []
            scores = []
            for line in result:
                if line[1][0][0] in num:
                    boxes.append(line[0])
                    txts.append(int(line[1][0]))
                    scores.append(line[1][1])
            # txts[0]为识别出来的文字
            txts = np.array(txts)
            if len(txts) > 0:
                shuzhi = np.max(txts)
            else:
                shuzhi = 0
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if h >= hmax:
                xmax, ymax, wmax, hmax = x, y, w, h
        # wmax和hmax为黑色区域最大框
        biaochi_length_h = hmax


    print(shuzhi)
    if biaochi_length_w >= biaochi_length_h:
        size = float(shuzhi) / biaochi_length_w
    else:
        size = float(shuzhi) / biaochi_length_h

    return size

#识别出门的中心点，主要思想是通过颜色框出门的矩形，并通过对矩形四个点周围像素的类别分析找到门的中心店
def finddoor(image_path, size):
    image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 设置深蓝色的阈值范围
    #lower_black = np.array([119, 128, 250], dtype=np.uint8)
    #upper_black = np.array([120, 255, 255], dtype=np.uint8)
    lower_black = np.array([110, 100, 200], dtype=np.uint8)
    upper_black = np.array([130, 255, 255], dtype=np.uint8)

    # 根据阈值范围创建黑色的掩码
    mask = cv2.inRange(hsv, lower_black, upper_black)

    # 查找轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    door = []
    #print(len(contours))
    #用于查看可视化结果
    show_img = image.copy()
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        points = np.array([[x, y], [x + w, y], [x, y + h], [x + w, y + h]])
        cv2.rectangle(show_img, (x,y), (x+w,y+h), (0,0,255),2)
        finalpoints = []
        for point in points:
            hwall, hdoor, hwindow, hnum = circlehascolor(image, point[0], point[1], 2)
            hwall2, hdoor2, hwindow2, hnum2 = circlehascolor(image, point[0], point[1], 3)
            #增加一个大半径的检测，以防小半径连门的颜色都检测不到
            if (hwall == 1 and hdoor2 == 1) or (hwindow == 1):
                finalpoints.append(point)
        #print(finalpoints)
        if len(finalpoints) == 4:
            cenx = int(x+w/2)
            ceny = int(y+h/2)
            door.append([cenx, ceny])
        # 处理一条样式的门,目前和下面的逻辑有冲突，但是要再测试发现问题再改
        elif len(finalpoints) == 1:
            cenx = int(x + w / 2)
            ceny = int(y + h / 2)
            door.append([cenx, ceny])
        #专门来解决墙的一侧离门太近导致三个点的情况
        elif len(finalpoints) == 3:
            two_center = []
            for i in range(len(finalpoints)):
                if i == len(finalpoints) -1:
                    point1 = finalpoints[i]
                    point2 = finalpoints[0]
                else:
                    point1 = finalpoints[i]
                    point2 = finalpoints[i+1]
                if abs(point1[0] - point2[0]) <= 5 or abs(point1[1] - point2[1]) <= 5:
                    cenx = int((point1[0] + point2[0]) / 2)
                    ceny = int((point1[1] + point2[1]) / 2)
                    two_center.append([cenx, ceny])
            print(two_center)
            #应该得到两个中心点，要那个周围什么都没有的中心点
            this_door = []
            if len(two_center) == 2:
                for data in two_center:
                    hwall, hdoor, hwindow, hnum = circlehascolor(image, data[0], data[1], 2)
                    if hwall == 0 and hdoor == 0:
                        this_door.append([data[0], data[1]])
                #一条样式的门也会有三个点的情况，到这里如果是一条样式的门就会得到空的结果，需要再处理一下
                if len(this_door) == 0:
                    cenx = int(x + w / 2)
                    ceny = int(y + h / 2)
                    door.append([cenx, ceny])
                else:
                    door.append(this_door[0])
        else:
            if len(finalpoints) < 2:
                finalpoints = []
                for point in points:
                    hwall, hdoor, hwindow, hnum = circlehascolor(image, point[0], point[1], 2)
                    if hwall == 1:
                        finalpoints.append(point)
            for i in range(len(finalpoints)-1):
                cenx = int((finalpoints[i][0]+finalpoints[i+1][0])/2)
                ceny = int((finalpoints[i][1] + finalpoints[i + 1][1]) / 2)
                door.append([cenx,ceny])
    # cv2.namedWindow("mask", cv2.WINDOW_NORMAL)
    # cv2.imshow('mask', mask)
    #print(door)
    # cv2.namedWindow("show", cv2.WINDOW_NORMAL)
    # cv2.imshow('show', show_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    cv2.imwrite("/az-data/FloorPlan-Server/demo/debug/door_mask.png", mask)
    return door

#该函数目前没有使用了，暂时保留防止以后使用
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
            if d*size <= 200:
                if i == len(poly) - 1:
                    index = [0]
                    poly = np.delete(poly, index, 0)
                else:
                    index = [i + 1]
                    poly = np.delete(poly, index, 0)
                break
                # condition b
            if angle < 10:
                if i == len(poly) -1:
                    index = [0]
                else:
                    index = [i + 1]
                poly = np.delete(poly, index, 0)
                break
            if i == len(poly) - 1:
                done = True
    return poly

#识别出窗的两个端点，主要思想类似找门的中心点，也是通过颜色框出窗的矩形，然后判断矩形四个点周围像素的类别
#有一点请注意，窗的形状有两中特殊的，L型的二连窗，还有三连窗，在识别出这两种窗时需要将其中代表墙点的点返回
def findwindow(image_path, size):
    image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # 设置浅蓝色的阈值范围
    lower_black = np.array([85, 200, 230], dtype=np.uint8)
    upper_black = np.array([92, 255, 255], dtype=np.uint8)
    # 根据阈值范围创建黑色的掩码
    mask = cv2.inRange(hsv, lower_black, upper_black)

    # 查找轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    window = []
    center = []
    addwall = np.array([])
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        four_points = np.array([[x,y], [x,y+h], [x+w, y+h], [x+w, y]])
        four_points_color = []
        #判断矩形四个点周围的像素类别
        for data in four_points:
            hascolor = list(circlehascolor(image, data[0], data[1], 5))
            four_points_color.append(hascolor)
        nowallindex = []
        for index, item in enumerate(four_points_color):
            #如果该点周围只有窗，就代表该点应该加入墙点中
            if item == [0,0,1,0]:
                nowallindex.append(index)
        if w*size <= 100 and h*size <= 100:
            continue
        if len(center) > 0:
            new_center = [int(x + w / 2), int(y + h / 2)]
            last_center = center[len(center) - 1]
            d = sqrt((last_center[0] - new_center[0]) ** 2 + (last_center[1] - new_center[1]) ** 2)
            if d * size <= 250:
                continue
            else:
                center.append([int(x + w / 2), int(y + h / 2)])
        else:
            center.append([int(x+w/2), int(y+h/2)])
        print("nowwallindex: ", nowallindex)
        #三个窗相连情况处理
        if len(nowallindex) == 2:
            nowallindex = []
            for index, item in enumerate(four_points_color):
                if item == [0, 0, 1, 0]:
                    nowallindex.append(index)
            print("三连窗情况:", nowallindex, four_points)
            #现改为将三连窗视为三个窗，则需要加入三条窗户边
            '''
            如果nowallindex是
            0,1，，，则加入3,0-0,1-1,2
            1,2，，，则加入0,1-1,2-2,3
            2,3，，，则加入1,2-2,3-3,0
            3,0，，，则加入2,3-3,0-0,1
            '''
            if nowallindex == [0,1]:
                window.append(np.array([four_points[3],  four_points[0]]))
                window.append(np.array([four_points[0],  four_points[1]]))
                window.append(np.array([four_points[1],  four_points[2]]))
            elif nowallindex == [1,2]:
                window.append(np.array([four_points[0],  four_points[1]]))
                window.append(np.array([four_points[1],  four_points[2]]))
                window.append(np.array([four_points[2],  four_points[3]]))
            elif nowallindex == [2,3]:
                window.append(np.array([four_points[1],  four_points[2]]))
                window.append(np.array([four_points[2],  four_points[3]]))
                window.append(np.array([four_points[3],  four_points[0]]))
            else:
                window.append(np.array([four_points[2],  four_points[3]]))
                window.append(np.array([four_points[3],  four_points[0]]))
                window.append(np.array([four_points[0],  four_points[1]]))
            if len(addwall) == 0:
                addwall = four_points
            else:
                addwall = np.concatenate((addwall, four_points))
        #两个窗相连情况处理
        elif len(nowallindex) == 1:
            if len(addwall) == 0:
                addwall = np.array([four_points[nowallindex[0]]])
            else:
                addwall = np.concatenate((addwall, np.array([four_points[nowallindex[0]]])))
            nowallindex = []
            for index, item in enumerate(four_points_color):
                if item == [1, 0, 1, 0]:
                    nowallindex.append(index)
            #TODO：两连窗也需要跟三连窗一样更改
            window.append(np.array([four_points[nowallindex[0]], four_points[nowallindex[1]]]))

        else:
            if w <= h:
                # cv2.circle(image, (int(x+w/2), y), 5, (0, 0, 255), -1)
                # cv2.circle(image, (int(x+w/2), y + h), 5, (0, 0, 255), -1)
                window.append(np.array([[int(x + w / 2), y], [int(x + w / 2), y + h]]))
            else:
                # cv2.circle(image, (x, int(y+h/2)), 5, (0, 0, 255), -1)
                # cv2.circle(image, (x + w, int(y+h/2)), 5, (0, 0, 255), -1)
                window.append(np.array([[x, int(y + h / 2)], [x + w, int(y + h / 2)]]))

    # cv2.namedWindow("mask", cv2.WINDOW_NORMAL)
    # cv2.imshow('mask', mask)
    # cv2.namedWindow("ori", cv2.WINDOW_NORMAL)
    # cv2.imshow('ori', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    #print("window********:",window)
    cv2.imwrite("/root/FloorPlan-Server/demo/debug/window_mask.png", mask)
    return window, addwall

#识别出框的开间进深
def calw_h(image_path, size, wall, door, window):
    image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
    rec_point = []
    for data in wall:
        rec_point.append([list(data)])
    for data in door:
        rec_point.append([list(data)])
    for data in window:
        rec_point.append([list(data[0])])
        rec_point.append([list(data[1])])
    x, y, w, h = cv2.boundingRect(np.array(rec_point))

    witdh = w*size
    height = h*size
    return witdh, height

#删除距离很近的点
def del_close_point(type, points, size):
    if type == "door" or type == "wall":
        done = False
        while not done:
            if len(points) <= 1:
                done = True
                break
            for i in range(len(points)):
                if i == len(points) - 1:
                    v1 = points[i]
                    v2 = points[0]
                else:
                    v1 = points[i]
                    v2 = points[i+1]
                if np.linalg.norm((v1-v2))*size <= 200:
                    if i == len(points) - 1:
                        index = [0]
                        points = np.delete(points, index, 0)
                    else:
                        index = [i + 1]
                        points = np.delete(points, index, 0)
                    break
                if i == len(points) - 1:
                    done = True
    return points


