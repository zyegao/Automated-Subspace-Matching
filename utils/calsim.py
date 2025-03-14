import numpy as np
import cv2
from shapely.geometry import Polygon
from shapely.affinity import rotate
import math

def new_calsim(roomtype, l1, l2, size1, size2, roomwitdh, roomheight, kuangwitdh, kuangheight, code):
    door1 = np.array(l1[0])
    window1 = np.array(l1[1])
    wall1 = np.array(l1[2])
    door2 = np.array(l2[0])
    window2 = np.array(l2[1])
    wall2 = np.array(l2[2])
 
    if len(wall2) == 0:
        return 0
    x1, y1, w1, h1 = cv2.boundingRect(wall1)
    center1 = np.array([int(x1 + w1 / 2), int(y1 + h1 / 2)])
    x2, y2, w2, h2 = cv2.boundingRect(wall2)
    center2 = np.array([int(x2 + w2 / 2), int(y2 + h2 / 2)])
    #print(code)
    #ratewall数组分别代表原图，原图镜像，90度旋转，90度旋转镜像，180旋转，180度旋转镜像，270度旋转，270度旋转镜像后的形状相似度，形状一致为0.2，不一致为零
    ratewall = np.array([0,0,0,0,0,0,0,0])
    #如果门窗数量不一致，直接返回
    if len(door1) != len(door2) or len(window1) != len(window2):
        return -1
    #求形状相似度，在墙点数量一下情况下才会去计算
    if len(wall1) == len(wall2):
        #ratewall = wall_rate(wall1,door1,window1, wall2,door2,window2,code, size1, size2)
        # 用作形状相似度对比，先不算上尺寸
        xh_wall1 = wall1
        xh_door1 = door1
        xh_window1 = window1

        xh_wall2 = wall2
        xh_door2 = door2
        xh_window2 = window2

        # 需要计算原图，90度，180度，270度旋转后的形状编码
        # 原图的数据进行赋值
        xh_wall2_0 = xh_wall2
        xh_door2_0 = xh_door2
        xh_window2_0 = xh_window2
        #print(xh_wall2, xh_door2, xh_window2)
        # 下面进行旋转后为90，180，270的赋值

        xh_window2 = window2.reshape(-1, 2)
        xh_window2_len = len(xh_window2)
        xh_door2_len = len(xh_door2)
        xh_alldata2 = xh_wall2
        if xh_window2_len > 0:
            xh_alldata2 = np.concatenate((xh_window2, xh_alldata2), axis=0)

        if xh_door2_len > 0:
            xh_alldata2 = np.concatenate((xh_door2, xh_alldata2), axis=0)
        xh_alldata2 = xh_alldata2 - center2
        xh_alldata2 = Polygon(xh_alldata2)
        #print(list(xh_alldata2.exterior.coords))
        xh_alldata2_90 = list(rotate(xh_alldata2, 90).exterior.coords)
        #print(xh_alldata2_90)
        xh_alldata2_180 = list(rotate(xh_alldata2, 180).exterior.coords)
        #print(xh_alldata2_180)
        xh_alldata2_270 = list(rotate(xh_alldata2, 270).exterior.coords)
        #print(xh_alldata2_270)



        xh_door2_90 = np.array(xh_alldata2_90[:xh_door2_len])
        xh_window2_90 = np.array(xh_alldata2_90[xh_door2_len:xh_door2_len + xh_window2_len])
        xh_window2_90 = xh_window2_90.reshape(int((xh_window2_len + 1) / 2), 2, 2)
        xh_wall2_90 = np.array(xh_alldata2_90[xh_door2_len + xh_window2_len:-1])

        xh_door2_180 = np.array(xh_alldata2_180[:xh_door2_len])
        xh_window2_180 = np.array(xh_alldata2_180[xh_door2_len:xh_door2_len + xh_window2_len])
        xh_window2_180 = xh_window2_180.reshape(int((xh_window2_len + 1) / 2), 2, 2)
        xh_wall2_180 = np.array(xh_alldata2_180[xh_door2_len + xh_window2_len:-1])

        xh_door2_270 = np.array(xh_alldata2_270[:xh_door2_len])
        xh_window2_270 = np.array(xh_alldata2_270[xh_door2_len:xh_door2_len + xh_window2_len])
        xh_window2_270 = xh_window2_270.reshape(int((xh_window2_len + 1) / 2), 2, 2)
        xh_wall2_270 = np.array(xh_alldata2_270[xh_door2_len + xh_window2_len:-1])

        wall1_code = encode_wall(xh_wall1, xh_door1, xh_window1, size1)

        wall2_0_code = encode_wall(xh_wall2_0, xh_door2_0, xh_window2_0, size2)
        wall2_0_code_reverse_index = np.array(list(range(len(wall2_0_code) - 1, -1, -1)))
        wall2_0_code_reverse = wall2_0_code[wall2_0_code_reverse_index]

        wall2_90_code = encode_wall(xh_wall2_90, xh_door2_90, xh_window2_90, size2)
        wall2_90_code_reverse_index = np.array(list(range(len(wall2_90_code) - 1, -1, -1)))
        wall2_90_code_reverse = wall2_90_code[wall2_90_code_reverse_index]

        wall2_180_code = encode_wall(xh_wall2_180, xh_door2_180, xh_window2_180, size2)
        wall2_180_code_reverse_index = np.array(list(range(len(wall2_180_code) - 1, -1, -1)))
        wall2_180_code_reverse = wall2_180_code[wall2_180_code_reverse_index]

        wall2_270_code = encode_wall(xh_wall2_270, xh_door2_270, xh_window2_270, size2)
        wall2_270_code_reverse_index = np.array(list(range(len(wall2_270_code) - 1, -1, -1)))
        wall2_270_code_reverse = wall2_270_code[wall2_270_code_reverse_index]

        wall2_all_code = np.array(
            [wall2_0_code, wall2_0_code_reverse, wall2_90_code, wall2_90_code_reverse, wall2_180_code,
             wall2_180_code_reverse, wall2_270_code, wall2_270_code_reverse])
        #print(wall2_all_code)
        #print(wall1_code)
        for i in range(len(wall2_all_code)):
            if len(wall2_all_code[i]) == len(wall1_code):
                if (np.array(wall2_all_code[i]) == np.array(wall1_code)).all():
                    ratewall[i] = 1

        #print(ratewall)

        # print("爆炸图编码: ", encode_wall(xh_wall1, xh_door1, xh_window1, size1))
        # print("案例原图编码: ", encode_wall(xh_wall2_0, xh_door2_0, xh_window2_0, size2))
        # print("案例90编码: ", encode_wall(xh_wall2_90, xh_door2_90, xh_window2_90, size2))
        # print("案例180编码: ", encode_wall(xh_wall2_180, xh_door2_180, xh_window2_180, size2))
        # print("案例270编码: ", encode_wall(xh_wall2_270, xh_door2_270, xh_window2_270, size2))




    #在计算后面的相似度时涉及到旋转镜像操作，因为我要考虑旋转后一致或者镜像一致的案例
    #对于那些门窗数量为零的，将其中心点给门窗，这样旋转镜像才能操作


    # 镜像矩阵
    flip_np = np.array([[-1, 0], [0, 1]])
    # 我发现有的案例是只有两面墙的，主要是餐厅，这导致墙点只有三个
    if len(wall2) <= 3:
        wall2 = np.array([[x2, y2], [x2 + w2, y2], [x2 + w2, y2 + h2], [x2, y2 + h2]])

    if len(door1) == 0:
        door1 = np.array([center1])
    if len(window1) == 0:
        window1 = np.array([[center1, center1]])
    if len(door2) == 0:
        door2 = np.array([center2])
    if len(window2) == 0:
        window2 = np.array([[center2, center2]])

    #将尺寸附给向量，后面的相似度计算才能考虑到尺寸
    newdoor1 = (door1 - center1)*size1

    len_door2 = len(door2)
    door2 = np.concatenate((door2,wall2), axis = 0)
    newdoor2 = Polygon((door2 - center2) * size2)
    center_door2 = (door2 - center2) * size2
    #镜像操作
    newdoor3 = np.dot(center_door2, flip_np)
    newdoor3 = Polygon(newdoor3)

    newwindow1 = (window1 - center1) * size1

    window2 = window2.reshape(-1,2)
    len_window2 = len(window2)

    window2 = np.concatenate((window2, wall2), axis = 0)
    newwindow2 = Polygon((window2 - center2) * size2)
    center_window2 = (window2 - center2) * size2
    #镜像操作
    newwindow3 = np.dot(center_window2, flip_np)
    newwindow3 = Polygon(newwindow3)



    newwall1 = Polygon((wall1 - center1) * size1)
    newwall2 = Polygon((wall2 - center2) * size2)
    center_wall2 = (wall2 - center2) * size2
    #镜像操作
    newwall3 = np.dot(center_wall2, flip_np)
    newwall3 = Polygon(newwall3)

    # 如果房间与框的长宽关系一致,只需旋转框180度,又增加了一个条件如果w和h相差很小很小几乎相等就不用旋转了
    # 在案例的原数据和镜像数据上进行旋转
    # 根据旋转与否选择需要加上的四个形状相似度值
    add_ratewall = np.array([0,0,0,0])
    if (w1 <= h1 and w2 <= h2) or (w1 >= h1 and w2 >= h2) or abs(h2-w2)*size2 <= 10:
        newdoor2_1 = newdoor2
        newdoor2_2 = rotate(newdoor2, 180)
        newdoor3_1 = newdoor3
        newdoor3_2 = rotate(newdoor3, 180)

        newwindow2_1 = newwindow2
        newwindow2_2 = rotate(newwindow2, 180)
        newwindow3_1 = newwindow3
        newwindow3_2 = rotate(newwindow3, 180)

        newwall2_1 = newwall2
        newwall2_2 = rotate(newwall2, 180)
        newwall3_1 = newwall3
        newwall3_2 = rotate(newwall3, 180)
        #原图，原图镜像，180，80镜像
        add_ratewall = np.array([ratewall[0], ratewall[1], ratewall[4], ratewall[5]])


    else:
        newdoor2_1 = rotate(newdoor2, 90)
        newdoor2_2 = rotate(newdoor2, 270)
        newdoor3_1 = rotate(newdoor3, 90)
        newdoor3_2 = rotate(newdoor3, 270)

        newwindow2_1 = rotate(newwindow2, 90)
        newwindow2_2 = rotate(newwindow2, 270)
        newwindow3_1 = rotate(newwindow3, 90)
        newwindow3_2 = rotate(newwindow3, 270)

        newwall2_1 = rotate(newwall2, 90)
        newwall2_2 = rotate(newwall2, 270)
        newwall3_1 = rotate(newwall3, 90)
        newwall3_2 = rotate(newwall3, 270)

        # 90，90镜像，270，270镜像
        add_ratewall = np.array([ratewall[2], ratewall[3], ratewall[6], ratewall[7]])


    #去求爆炸图数据与四个经过旋转镜像操作后的案例数据的两个指标相似度
    rateiou1 = iou_rate2(newwall1, newwall2_1)
    rateiou2 = iou_rate2(newwall1, newwall2_2)
    rateiou3 = iou_rate2(newwall1, newwall3_1)
    rateiou4 = iou_rate2(newwall1, newwall3_2)
    #print(rateiou1, rateiou2, rateiou3, rateiou4)
    ratewindow1 = window_door_rate(newdoor1, newdoor2_1, newwindow1, newwindow2_1, len_door2, len_window2)
    ratewindow2 = window_door_rate(newdoor1, newdoor2_2, newwindow1, newwindow2_2, len_door2, len_window2)
    ratewindow3 = window_door_rate(newdoor1, newdoor3_1, newwindow1, newwindow3_1, len_door2, len_window2)
    ratewindow4 = window_door_rate(newdoor1, newdoor3_2, newwindow1, newwindow3_2, len_door2, len_window2)

    #可为两个指标设置权重
    theta_iou = 0.5
    theta_window = 0.5

    totalrate = np.array([theta_iou*rateiou1 + theta_window*ratewindow1, theta_iou*rateiou2 + theta_window*ratewindow2, theta_iou*rateiou3 + theta_window*ratewindow3, theta_iou*rateiou4 + theta_window*ratewindow4])
    #print("*****************************************************************************************************")
    #print([rateiou1, rateiou2, rateiou3, rateiou4], [ratewindow1, ratewindow2, ratewindow3, ratewindow4], totalrate)
    #获得两个指标结合后最高的相似度
    #加上形状相似度，如果形状一直，会极大的提高最后的相似度，来确保返回形状一致的那个，及时形状一致的这个尺寸不好
    totalrate = totalrate + add_ratewall

    max_totalrate = max(totalrate)
    #print(code, max_totalrate, ratewall)

    return max_totalrate
#计算点到线中点的距离
def get_distance_from_point_to_line(point, line_point1, line_point2):
    #对于两点坐标为同一点时,返回点与点的距离
    if (line_point1 == line_point2).all():
        point_array = np.array(point )
        point1_array = np.array(line_point1)
        return np.linalg.norm(point_array -point1_array )
    #计算直线的三个参数
    A = line_point2[1] - line_point1[1]
    B = line_point1[0] - line_point2[0]
    C = (line_point1[1] - line_point2[1]) * line_point1[0] + (line_point2[0] - line_point1[0]) * line_point1[1]
    #根据点到直线的距离公式计算距离
    distance = np.abs(A * point[0] + B * point[1] + C) / (np.sqrt(A**2 + B**2))
    # middlepoint = np.array([(line_point1[0]+line_point2[0])/2, (line_point1[1] + line_point2[1])/2])
    # distance = np.linalg.norm(np.array(point) - middlepoint)
    return distance

#判断点到线上的投影是否在线上
def point_on_line(point, line_point1, line_point2):
    px, py = point[0], point[1]
    x1, y1 = line_point1[0], line_point1[1]
    x2, y2 = line_point2[0], line_point2[1]

    line_magnitude = math.sqrt(math.pow((x2 - x1), 2) + math.pow((y2 - y1), 2))
    u1 = (((px - x1) * (x2 - x1)) + ((py - y1) * (y2 - y1)))  # 向量AB·向量AP
    u = u1 / (line_magnitude * line_magnitude)  # 向量AP在向量AB方向的投影与向量AB模的比值
    if (u < 0) or (u > 1):
        # 点到直线的投影不在线段内
        return False
    else:
        return True

    return True

#利用上面函数中的方法得到爆炸图或者案例的形状编码
def encode_wall(wall,door,window,size):

    #所有墙点原始情况下是逆时针排序的，我们转换为顺时针排序的
    wall_reverse = np.array(list(range(len(wall) - 1, -1, -1)))
    wall = wall[wall_reverse]
    first_wall = np.array(sorted(wall, key=lambda x: (x[0], x[1])))[0]
    #print(first_wall)
    first_wall_index = 0
    for i in range(len(wall)):
        if (wall[i] == first_wall).all():
            first_wall_index = i
            break
    #first_wall_index = np.where(wall == first_wall[0], first_wall[1])
    #print(first_wall, first_wall_index)

    wall = np.concatenate((wall[first_wall_index:], wall[:first_wall_index]), axis= 0)

    #print(wall, door, window)
    # wall的点都是顺时针排序的了
    one_hot_wall = []

    one_hot_dir_wall = []

    one_hot_dw = np.zeros(len(wall))

    doorindex = -1

    windowindex = -1
    if len(door) > 0:
        for l in range(len(door)):
            min_d_w = 20000
            min_i = 0
            point_in_line = False
            for i in range(len(wall)):
                if i == len(wall) - 1:
                    dis = get_distance_from_point_to_line(door[l], wall[i], wall[0]) * size
                    point_in_line = point_on_line(door[l], wall[i], wall[0])
                else:
                    dis = get_distance_from_point_to_line(door[l], wall[i], wall[i + 1]) * size
                    point_in_line = point_on_line(door[l], wall[i], wall[i + 1])

                if dis < min_d_w and point_in_line:
                    min_d_w = dis
                    min_i = i

            doorindex = min_i


            one_hot_dw[min_i] += 8  # 门和床加的要不一样，而且要岔开，防止门窗加完一样


    # print(one_hot_dw1, one_hot_dw2, code)
    if len(window) > 0:
        for l in range(len(window)):
            min_d_w = 10000
            min_i = 0
            point_in_line = False
            window_point = np.array(
                [int((window[l][0][0] + window[l][1][0]) / 2), int((window[l][0][1] + window[l][1][1]) / 2)])
            for i in range(len(wall)):
                if i == len(wall) - 1:
                    dis = get_distance_from_point_to_line(window_point, wall[i], wall[0])*size
                    point_in_line = point_on_line(window_point, wall[i], wall[0])
                else:
                    dis = get_distance_from_point_to_line(window_point, wall[i], wall[i + 1])*size
                    point_in_line = point_on_line(window_point, wall[i], wall[i + 1])
                if dis <= min_d_w and point_in_line:
                    min_d_w = dis
                    min_i = i

            windowindex = min_i
            one_hot_dw[min_i] += 4  # 门和床加的要不一样，而且要岔开，防止门窗加完一样


    for i in range(len(wall)):
        if i == len(wall) - 1:
            v1 = np.array(wall[0]) - np.array(wall[i])
        else:
            v1 = np.array(wall[i + 1]) - np.array(wall[i])

        x = np.array([0, 1])
        Lx = np.sqrt(x.dot(x))
        y = np.array([1, 0])
        Ly = np.sqrt(y.dot(y))

        Lv1 = np.sqrt(v1.dot(v1))
        cos_angle_x1 = np.arccos(x.dot(v1) / (Lx * Lv1)) * 360 / 2 / np.pi
        cos_angle_x3 = np.arccos(y.dot(v1) / (Ly * Lv1)) * 360 / 2 / np.pi


        if abs(90 - cos_angle_x1) > 20:  # 上或者下
            if abs(180 - cos_angle_x1) > 10:  # 下，为-1
                one_hot_dir_wall.append(-1)
            else:
                one_hot_dir_wall.append(1)  # 上为1
            # one_hot_wall1.append(0)
        else:  # 左或右
            if abs(180 - cos_angle_x3) > 10:  # 右为-2
                one_hot_dir_wall.append(-2)
            else:
                one_hot_dir_wall.append(2)  # 左为2

    # 270度的在下面列表中，因为要判断出该角是90度还是270度，我们的点是顺时针的，所以可以通过与该角想连的两个向量的走向来判断
    dunjiao = [[-2, 1], [2, -1], [-1, -2], [1, 2]]
    for i in range(len(one_hot_dir_wall)):
        if i == 0:
            singlejiao1 = [one_hot_dir_wall[len(one_hot_dir_wall)-1], one_hot_dir_wall[0]]

        else:
            singlejiao1 = [one_hot_dir_wall[i - 1], one_hot_dir_wall[i]]

        if singlejiao1 in dunjiao:
            one_hot_wall.append(1)
        else:
            one_hot_wall.append(0)

    one_hot_wall = np.array(one_hot_wall)

    # 以下过程将门窗信息加入到墙的编码中，注意第一个墙编码数组以左上方的点为开始（x坐标最小中y坐标最小的点）
    insert_num = 0
    for i in range(len(one_hot_dw)):
        if one_hot_dw[i] > 0:
            if i == len(one_hot_wall)-1:
                one_hot_wall = np.append(one_hot_wall, one_hot_dw[i])
            else:
                one_hot_wall = np.insert(one_hot_wall, i+1+insert_num, one_hot_dw[i], axis=0)
                insert_num += 1
    return one_hot_wall

#计算门窗相对关系的相似度，参考了人脸识别
def window_door_rate(door1, door2, window1, window2, len_door, len_window):
    windowsum = 0
    door2 = np.array(list(door2.exterior.coords)[:len_door])
    window2 = np.array(list(window2.exterior.coords)[:len_window])
    window2 = window2.reshape(int((len_window+1)/2),2,2)
    for i in range(len(window1)):
        for j in range(len(door1)):
            window1[i] -= door1[j]
            window2[i] -= door2[j]

    for i in range(len(window1)):
        windowsum += np.linalg.norm(window1[i] - window2[i])
    ratewindow = 1 - np.tanh(windowsum / 10000)

    return ratewindow

##计算爆炸图和案例在考虑尺寸下，形状的相似度。可理解为二者的重合面积，目前用的是这个
def iou_rate2(polypon1, polypon2):
    try:
        inter_area1 = polypon1.intersection(polypon2).area
        union_area1 = polypon1.area + polypon2.area - inter_area1
        iou1 = float(inter_area1) / union_area1

    except Exception:
        print("wxp")
        return 0

    return iou1
