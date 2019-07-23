# 编程者：孙克染
# 功能：实现计算pm2.5预测栅格
# 输入：回归系数栅格、自变量栅格
# 输出：pm2.5预测栅格

import numpy as np
from osgeo import gdal
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


class Grid(object):
    @staticmethod
    def read_img(_file):
        dataset = gdal.Open(_file)
        # 数据描述
        # print(dataset.GetDescription())

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
        # print(img_data.shape)

        del dataset
        return img_data, img_proj, img_geotrans

    @staticmethod
    def write_img(_file, img_data, img_proj, img_geotrans, _format):
        # 判断栅格数据的数据类型
        if 'int8' in img_data.dtype.name:
            datatype = gdal.GDT_Byte
        elif 'int16' in img_data.dtype.name:
            datatype = gdal.GDT_UInt16
        else:
            datatype = gdal.GDT_Float64

        # 判读数组维数
        if len(img_data.shape) == 3:
            img_bands, img_height, img_width = img_data.shape
        else:
            img_bands, (img_height, img_width) = 1, img_data.shape

        # 创建文件
        # HFA -> .img | GTiff -> .tif
        if _format == 'tif':
            driver = gdal.GetDriverByName("GTiff")
        else:
            driver = gdal.GetDriverByName("HFA")

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


parser = ArgumentParser("aod_retrieval", formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler='resolve')
parser.add_argument('--date')
parser.add_argument('--utm')
args = parser.parse_args()
date_str = args.date
utm_str = args.utm
line_num, row_num = 2649, 2449

if __name__ == '__main__':
    print('PM2.5 calculation starts!')
    data = dict()
    dic1 = './img_data/'
    dic2 = '-' + date_str + '-' + utm_str + '.tif'
    cs = ['intercept', 'aod', 't', 'p', 'ws', 'rh', 'dem', 'ndvi']
    for index in range(len(cs)):
        dic = './result/c_' + cs[index] + dic2
        img_temp, proj_temp, geotrans_temp = Grid.read_img(dic)
        data['c_' + cs[index]] = img_temp
    for index in range(1, len(cs)):
        if index < len(cs) - 2:
            dic = dic1 + cs[index] + dic2
        else:
            dic = dic1 + cs[index] + '.tif'
        img_temp, proj_temp, geotrans_temp = Grid.read_img(dic)
        data[cs[index]] = img_temp
        data['proj'] = proj_temp
        data['geo'] = geotrans_temp
    img_result = data['c_intercept'] + data['c_aod'] * data['aod'] + data['c_t'] * data['t'] + \
                 data['c_p'] * data['p'] + data['c_ws'] * data['ws'] + data['c_rh'] * data['rh'] + \
                 data['c_dem'] * data['dem'] + data['c_ndvi'] * data['ndvi']
    Grid.write_img('./result/pm25_predict-' + date_str + '-' + utm_str + '.tif',
                   img_result, data['proj'], data['geo'], 'tif')
    print('PM2.5 calculation finished!')
