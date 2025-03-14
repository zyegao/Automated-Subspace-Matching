from paddleocr import PaddleOCR, draw_ocr
import numpy as np
#删除数组中方差最大的那个元素
def remove_variance_max(arr):
    if len(arr) == 1:
        return arr
    variances = [(np.var(arr, ddof=1), i) for i, _ in enumerate(arr)]
    max_variance_index = max(variances, key=lambda x: x[0])[1]

    arr = np.delete(arr, max_variance_index)
    return arr
#识别户型图的尺寸信息
def calsize(img_path):
    ocr = PaddleOCR(use_angle_cls=False, lang="ch")  # need to run only once to download and load model into memory
    result = ocr.ocr(img_path, cls=True)
    boxes = []
    txts = []
    scores = []
    viald = []
    num = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    result = result[0]
    for line in result:
        allin = True
        for data in line[1][0]:
            #正常的尺寸信息应该都是数字，这里通过判断识别出来的字符是否在num数组里来判断该字符是不是数字
            if data in num:
                allin = True
            else:
                #一旦有一个不是数字，就不要这组字符
                allin = False
                break
        # 如果识别出来的第一位为0，则说明不对，不加进数组里来
        # 识别出来的数字过长可能是因为数字在图片中连在一起了，这种的就不要了
        if len(line[1][0]) > 4 or line[1][0][0] == "0":
            allin = False
        boxes.append(line[0])
        txts.append(line[1][0])
        scores.append(line[1][1])
        viald.append(allin)

    centerocr = []
    for data in boxes:
        centerocr.append([int((data[0][0] + data[1][0]) / 2), int((data[0][1] + data[2][1]) / 2)])
    size = []
    #得到单位像素的逻辑是，用两组想连的数字和除以2在除以该两组相连数字中心坐标的像素值差
    for i in range(len(centerocr) - 1):
        if abs(centerocr[i][1] - centerocr[i + 1][1]) <= 10 and viald[i] == True and viald[i+1] == True:
            print(txts[i], txts[i + 1])
            size.append((float(txts[i]) + float(txts[i + 1])) / 2 / (centerocr[i + 1][0] - centerocr[i][0]))
    size = np.array(size)
    if len(size) == 0:
        return None


    size = remove_variance_max(size)
    size = np.mean(size)
    return size



