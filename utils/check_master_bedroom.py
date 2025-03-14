#检查返回结果中是否有主卧，如果没有主卧，将面积最大的次卧变成主卧
import numpy as np
import math
import json
def check_master_bedroom(allbaozhame):
    bedroom_index = []
    bedroom_wh = []
    for i in range(len(allbaozhame)):
        print(allbaozhame[i]["roomname"])
        if allbaozhame[i]["roomname"] == "主卧":
            return allbaozhame
        if allbaozhame[i]["roomname"] == "次卧":
            bedroom_index.append(i)
            bedroom_wh.append([allbaozhame[i]["roomData"][0]["kaijian"], allbaozhame[i]["roomData"][0]["jinshen"]])
    
    max_index = find_max_bedroom(bedroom_index, bedroom_wh)
    #print(max_index, bedroom_index)
    allbaozhame[bedroom_index[max_index]]["roomname"] = "主卧"
    #print(allbaozhame)
    return allbaozhame
def find_max_bedroom(bedroom_index, bedroom_wh):
    bedroom_area = []
    for i in range(len(bedroom_wh)):
        w = float(bedroom_wh[i][0])
        h = float(bedroom_wh[i][1])
        bedroom_area.append(w*h)
    index = bedroom_area.index(max(bedroom_area))
    return index

