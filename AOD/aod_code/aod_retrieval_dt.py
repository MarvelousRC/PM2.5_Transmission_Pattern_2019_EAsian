import cv2
from osgeo import gdal
import os
import numpy as np
import matplotlib.pyplot as plt
import time
from argparse import ArgumentParser, FileType, ArgumentDefaultsHelpFormatter
import sys

# np.set_printoptions(threshold=np.inf)

__author__ = "Andrian Lee"
__date__ = "2019/01/07"

class Grid():
    def read_img(self, _file):
        dataset = gdal.Open(_file)
        # 数据描述
        print(dataset.GetDescription())

        # 图像的列数X与行数Y
        img_width = dataset.RasterXSize
        img_height = dataset.RasterYSize

        # 仿射矩阵
        img_geotrans = dataset.GetGeoTransform()

        # 投影
        img_proj = dataset.GetProjection()

        # 将数据写成数组，对应栅格矩阵
        img_data = dataset.ReadAsArray(0, 0, img_width, img_height)

        # 数据格式大小
        print(img_data.shape)

        del dataset
        return img_data, img_proj, img_geotrans

    def write_img(self, _file, img_data, img_proj, img_geotrans):
        # 判断栅格数据的数据类型
        if 'int8' in img_data.dtype.name:

            datatype = gdal.GDT_Byte
        elif 'int16' in img_data.dtype.name:
            datatype = gdal.GDT_UInt16
        else:
            datatype = gdal.GDT_Float32

        # 判读数组维数
        if len(img_data.shape) == 3:
            img_bands, img_height, img_width = img_data.shape
        else:
            img_bands, (img_height, img_width) = 1, img_data.shape

        # 创建文件
        driver = gdal.GetDriverByName("HFA")  # HFA -> .img | GTiff -> .tif
        dataset = driver.Create(_file, img_width, img_height, img_bands, datatype)

        # 写入仿射变换参数
        dataset.SetGeoTransform(img_geotrans)
        # 写入投影
        dataset.SetProjection(img_proj)
        # 写入数组数据
        # GetRasterBand()
        if img_bands == 1:
            dataset.GetRasterBand(1).WriteArray(img_data)
        else:
            for i in range(img_bands):
                dataset.GetRasterBand(i + 1).WriteArray(img_data[i])

        del dataset


def AOD_darktarget(data, _file, _method):
    lut = LUT_read(_file)
    row, col = data.shape[1], data.shape[2]
    aod = np.zeros((row, col))

    data, cnt_p = choose_darktarget(data)

    tot = row * col
    print("To be processed", cnt_p, "/", tot)

    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    # look up table
    for idi, i in enumerate(data[5]):
        for idj, j in enumerate(i):
            if j != 0.0 and j != -1.0:  # ignore mask and nodata
                aod_v = LUT_match(data, lut, idi, idj, _method)
                aod[idi][idj] = aod_v
        if idi % 50 == 0:
            print(idi)
            print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    return aod


def LUT_read(_file):
    f = open(_file)
    lines = f.readlines()
    f.close()
    res = []
    for line in lines:
        line = [float(x) for idx, x in enumerate(line.strip().split()) if idx in [0, 1, 2, 6]]
        res.append(line)
    return res


def choose_darktarget(data):
    """
        nodata -> -1.0
        dark target -> 0.0
    """
    nodata = data[0][0][0]
    ndvi = (data[1] - data[0]) / (data[1] + data[0])

    cnt = 0  # count pixels to process
    # mask -> 0.0
    for idi, i in enumerate(data[5]):
        for idj, j in enumerate(i):
            # process nan
            if j == nodata:
                data[0][idi][idj] = data[1][idi][idj] = data[5][idi][idj] = -1.0
            # process dark target
            elif j > 0.25 or ndvi[idi][idj] < 0.0:
                data[0][idi][idj] = data[1][idi][idj] = data[5][idi][idj] = 0.0
            else:
                cnt += 1

    return data, cnt


def LUT_match(data, lut, idi, idj, _method):
    if _method == 'r':
        p_aot1 = data[0][idi][idj]  # red
        p_sur = data[5][idi][idj] / 2  # red
    else:
        p_aot1 = data[2][idi][idj]  # blue
        p_sur = data[5][idi][idj] / 4  # blue

    min_d = 100
    aod_v = 0
    for i in lut:
        p_aot2 = i[2] + (i[1] * p_sur) / (1 - i[0] * p_sur)
        _delta = abs(p_aot2 - p_aot1)
        if _delta < min_d:
            aod_v = i[3]
            min_d = _delta

    return aod_v


def Reclassify(aod, data):
    """reclassify mask -> 0.0"""
    row, col = data.shape[1], data.shape[2]
    for idi, i in enumerate(data[0]):
        for idj, j in enumerate(i):
            if j == 0.0:
                x = 1
                while True:
                    flag = 0
                    for t in range(idj-x, idj+x+1):
                        if idi-x >= 0 and idi+x < row and 0 <= t < col:
                            if aod[idi-x][t] > 0.0 or aod[idi+x][t] > 0.0:
                                flag = 1
                                break
                    for t in range(idi-x, idi+x+1):
                        if 0 <= t < row and idj-x >= 0 and idj+x < col:
                            if aod[t][idj-x] > 0.0 or aod[t][idj+x] > 0.0:
                                flag = 1
                                break
                    if flag == 1:
                        break
                    x += 1

                cnt = 0
                tot = 0
                for t in range(idj-x, idj+x+1):
                    if idi - x >= 0 and idi + x < row and 0 <= t < col:
                        if aod[idi + x][t] >= 0.0:
                            tot += aod[idi + x][t]
                            cnt += 1
                        if aod[idi - x][t] >= 0.0:
                            tot += aod[idi - x][t]
                            cnt += 1
                for t in range(idi-x+1, idi+x):
                    if 0 <= t < row and idj - x >= 0 and idj + x < col:
                        if aod[t][idj - x] >= 0.0:
                            tot += aod[t][idj - x]
                            cnt += 1
                        if aod[t][idj + x] >= 0.0:
                            tot += aod[t][idj + x]
                            cnt += 1
                if cnt == 0:
                    print("srtange:", idi, " ", idj)
                    print("x", x)
                else:
                    aod[idi][idj] = tot / cnt
    return aod


def process(args):
    # read .img
    run = Grid()
    filename = args.input
    data, proj, geotrans = run.read_img(filename)
    # MODIS style
    # img_r, img_ir, img_g, img_b = data[0], data[1], data[3], data[2]

    lut_file = 'modis_lut_red.txt'
    if args.method == 'b':
        lut_file = 'modis_lut_blue.txt'

    out_data0 = AOD_darktarget(data, lut_file, args.method)

    out_data = Reclassify(out_data0, data)

    # plot and write .img
    plt.imshow(out_data)
    plt.show()
    out_file = filename[:-4] + '_aod_' + args.method + '.img'
    run.write_img(out_file, out_data, proj, geotrans)


def main():
    parser = ArgumentParser("aod_retrival", formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler='resolve')
    parser.add_argument('--input')
    parser.add_argument('--method')
    args = parser.parse_args()
    process(args)


if __name__ == "__main__":
    sys.exit(main())
