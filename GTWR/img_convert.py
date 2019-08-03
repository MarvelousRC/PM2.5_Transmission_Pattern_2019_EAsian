# 编程者：孙克染
# 功能：像元位深度修改为uint16，同时去除负值
# 输入：GWR预测pm25栅格
# 输出：uint16栅格

import numpy as np
from osgeo import gdal


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
            datatype = gdal.GDT_Float32

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


if __name__ == '__main__':
    date = 16
    utc = 7
    file_name = './pm25_predict_float32/pm25_predict-' + str(date) + '-' + str(utc) + '.tif'
    img_templete, proj_templete, geotrans_templete = Grid.read_img(file_name)
    row_num, column_num = img_templete.shape
    img_result = np.array([0] * (row_num * column_num), dtype='int16').reshape(row_num, column_num)
    print('Processing starts!')
    for i in range(row_num):
        for j in range(column_num):
            if img_templete[i, j] <= 0 or img_templete[i, j] == float("inf") or img_templete[i, j] >= 65535:
                img_result[i, j] = 0
            else:
                img_result[i, j] = int(img_templete[i, j] + 0.5)
        x = int(i / row_num * 100) + 1
        print('\r' + '▇' * (x // 2) + str(x) + '%', end='')
    out_file_name = './pm25_predict_uint16/pm25_predict_int-' + str(date) + '-' + str(utc) + '.tif'
    Grid.write_img(out_file_name, img_result, proj_templete, geotrans_templete, '.tif')
    print('\n' + str(date) + '-' + str(utc) + ' has been finished!')
