# 编程者：孙克染
# 功能：实现时空加权回归模型
# 输入：训练样本点CSV文件，预测点解释变量栅格
# 输出：回归系数栅格、local r^2栅格、模型拟合评价

import math
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
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


# global variables
# global y_avg, y_s2, x_min, x_max, y_min, y_max, b_final, img_templete, proj_templete, geotrans_templete
# global img_intercept, img_aod, img_t, img_p, img_ws, img_dem, img_ndvi
parser = ArgumentParser("aod_retrieval", formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler='resolve')
parser.add_argument('--start_date')
parser.add_argument('--end_date')
parser.add_argument('--hour_interval')
args = parser.parse_args()
start_date = int(args.start_date)
end_date = int(args.end_date)
hour_interval = int(args.hour_interval)
# start_date = 14
# end_date = 16
# hour_interval = 3
y_avg = 0  # 训练样本被解释变量平均值
y_s2 = 0  # 训练样本被解释变量方差
x_min = -1225790
x_max = 53952
y_min = -639572
y_max = 685124
b_final = 500000  # 确定的最佳带宽
b_n_final = 30  # 确定的最佳数量
file_name = './img_data/aod-14-1.tif'
list_train_y_predict = []
img_templete, proj_templete, geotrans_templete = Grid.read_img(file_name)
line_num, row_num = img_templete.shape
img_intercept = img_templete.copy()
img_aod = img_templete.copy()
img_t = img_templete.copy()
img_p = img_templete.copy()
img_ws = img_templete.copy()
img_rh = img_templete.copy()
img_dem = img_templete.copy()
img_ndvi = img_templete.copy()
img_local_r = img_templete.copy()
text_str = ''


def read_csv_file():
    source_data_l = {'id': [], 'name': [], 'lat': [], 'lon': [], 'pm2_5': [], 'aod': [], 't': [], 'p': [], 'ws': [],
                     'rh': [], 'dem': [], 'ndvi': [], 'x': [], 'y': [], 'time': []}
    with open(r'.\table\final_data.csv', 'r') as file:
        lines = file.read().splitlines()
        for i in range(len(lines)):
            lines[i].replace(' ', '')
            line_list = lines[i].split(',')
            if i != 0:
                source_data_l['id'].append(int(line_list[0]))
                source_data_l['name'].append(line_list[1])
                source_data_l['lon'].append(float(line_list[3]))
                source_data_l['lat'].append(float(line_list[2]))
                source_data_l['x'].append(float(line_list[14]))
                source_data_l['y'].append(float(line_list[15]))
                source_data_l['pm2_5'].append(float(line_list[4]))
                source_data_l['aod'].append(float(line_list[5]))
                source_data_l['t'].append(float(line_list[6]))
                source_data_l['p'].append(float(line_list[9]))
                source_data_l['ws'].append(float(line_list[11]))
                source_data_l['rh'].append(float(line_list[8]))
                source_data_l['dem'].append(float(line_list[13]))
                source_data_l['ndvi'].append(float(line_list[12]))
                source_data_l['time'].append(float(line_list[16]))
        return source_data_l


source_data = read_csv_file()
NUMBER = len(source_data['name'])  # 训练样本的个数
k = 7  # 解释变量的个数


def cal_weight_matrix(x, y, t, u, v, w, b_x, bandwidth_type, weight_matrix_type, number):
    factor = 10000
    result = np.eye(number)
    distance_list = []
    if bandwidth_type == 'fixed':
        if weight_matrix_type == 'bi-square':
            for i in range(number):
                d = math.sqrt((x - u[i]) ** 2 + (y - v[i]) ** 2 + ((t - w[i]) * factor) ** 2)
                if d <= b_x:
                    result[i][i] = (1 - (d / b_x) ** 2) ** 2
                else:
                    result[i][i] = 0.0
        else:
            for i in range(number):
                d = math.sqrt((x - u[i]) ** 2 + (y - v[i]) ** 2 + ((t - w[i]) * factor) ** 2)
                if d <= b_x:
                    result[i][i] = math.e ** (-(d / b_x) ** 2)
                else:
                    result[i][i] = 0.0
    else:
        for i in range(number):
            d = math.sqrt((x - u[i]) ** 2 + (y - v[i]) ** 2 + ((t - w[i]) * factor) ** 2)
            distance_list.append(d)
        distance_list.sort()
        b = distance_list[b_x - 1]
        if weight_matrix_type == 'bi-square':
            for i in range(number):
                d = math.sqrt((x - u[i]) ** 2 + (y - v[i]) ** 2 + ((t - w[i]) * factor) ** 2)
                if d <= b:
                    result[i][i] = (1 - (d / b) ** 2) ** 2
                else:
                    result[i][i] = 0.0
        else:
            for i in range(number):
                d = math.sqrt((x - u[i]) ** 2 + (y - v[i]) ** 2)
                if d <= b:
                    result[i][i] = math.e ** (-(d / b) ** 2)
                else:
                    result[i][i] = 0.0
    return result


def cal_result(matrix_w, matrix_xt_cal, matrix_x_cal, matrix_y_cal):
    temp = np.dot(matrix_xt_cal, matrix_w)
    temp1 = np.dot(temp, matrix_x_cal)
    global k
    if np.linalg.det(temp1) == 0:
        for i in range(k+1):
            temp1[i, i] += 0.000001
    temp3 = np.linalg.inv(temp1)
    temp2 = np.dot(temp, matrix_y_cal)
    matrix_b = np.dot(temp3, temp2)
    return matrix_b


def aic_test(matrix_xt_aic, matrix_x_aic, matrix_y_aic):
    global source_data, NUMBER, k, y_s2, b_final, b_n_final, list_train_y_predict
    # list_b = [1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000, 2000000, 5000000, 10000000,
    #           20000000, 50000000]
    list_b_n = list(range(2, 60))
    list_aic = []
    # 遍历每一种带宽
    for i in range(len(list_b_n)):
        # if list_b_n[i] == 30:
        #     print(list_b_n[i])
        #     print('实际值', '预测值')
        square_sum = 0
        list_y = []
        for j in range(NUMBER):
            # matrix_w = cal_weight_matrix(source_data['x'][j], source_data['y'][j], source_data['x'], source_data['y'],
            #                              list_b[i], 'fixed', 'bi-square', NUMBER)
            matrix_w = cal_weight_matrix(source_data['x'][j], source_data['y'][j], source_data['time'][j],
                                         source_data['x'], source_data['y'], source_data['time'],
                                         list_b_n[i], 'adaptive', 'bi-square', NUMBER)
            # print('权重矩阵 W: \n{}'.format(matrix_w))
            test_matrix_b = cal_result(matrix_w, matrix_xt_aic, matrix_x_aic, matrix_y_aic)
            # print('回归系数矩阵 b: \n{}'.format(test_matrix_b))
            y_pre = cal_predict(test_matrix_b, matrix_x_aic[j])
            square_sum += (y_pre - matrix_y_aic[j]) ** 2
            # if list_b_n[i] == 30:
            #     print(matrix_y_aic[j, 0], y_pre)
            list_y.append(y_pre)
        if list_b_n[i] == b_n_final:
            list_train_y_predict = list_y.copy()
        aic = math.log(square_sum / NUMBER) + 2 * (k + 1) / NUMBER
        list_aic.append(aic)
        # print('b: {}   r^2: {} {}'.format(list_b[i], test_global_r(list_y), 1-(square_sum / y_s2)))
        # print('b_n: {}   r^2: {:.4f} {:.4f}'.format(list_b_n[i], test_global_r(list_y), 1 - (square_sum / y_s2)))
    return list_aic


def aic_test_random():
    global source_data, NUMBER, text_str
    number_random = int(0.7 * NUMBER)
    # list_b = [1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000, 2000000, 5000000, 10000000,
    #           20000000, 50000000]
    list_b_n = list(range(1, int(0.69*NUMBER)))
    list_aic = []
    index_shuffle = list(range(NUMBER))
    random.shuffle(index_shuffle)
    source_data_random = {'name': [], 'lon': [], 'lat': [], 'pm2_5': [], 'aod': [], 't': [], 'p': [], 'ws': [],
                          'rh': [],  'dem': [], 'ndvi': [], 'x': [], 'y': [], 'time': []}
    for i in range(number_random):
        source_data_random['name'].append(source_data['name'][index_shuffle[i]])
        source_data_random['lon'].append(source_data['lon'][index_shuffle[i]])
        source_data_random['lat'].append(source_data['lat'][index_shuffle[i]])
        source_data_random['pm2_5'].append(source_data['pm2_5'][index_shuffle[i]])
        source_data_random['aod'].append(source_data['aod'][index_shuffle[i]])
        source_data_random['t'].append(source_data['t'][index_shuffle[i]])
        source_data_random['p'].append(source_data['p'][index_shuffle[i]])
        source_data_random['ws'].append(source_data['ws'][index_shuffle[i]])
        source_data_random['rh'].append(source_data['rh'][index_shuffle[i]])
        source_data_random['dem'].append(source_data['dem'][index_shuffle[i]])
        source_data_random['ndvi'].append(source_data['ndvi'][index_shuffle[i]])
        source_data_random['x'].append(source_data['x'][index_shuffle[i]])
        source_data_random['y'].append(source_data['y'][index_shuffle[i]])
        source_data_random['time'].append(source_data['time'][index_shuffle[i]])
    matrix_xt_random = np.mat([[1]*number_random, source_data_random['aod'], source_data_random['t'],
                               source_data_random['p'], source_data_random['ws'], source_data_random['rh'],
                               source_data_random['dem'], source_data_random['ndvi']])  # 训练集解释变量矩阵
    matrix_x_random = matrix_xt_random.T  # 训练集解释变量矩阵转置
    matrix_yt_random = np.mat(source_data_random['pm2_5'])  # 训练集被解释变量矩阵转置
    matrix_y_random = matrix_yt_random.T  # 训练集被解释变量矩阵
    text_str += '\n不同的带宽数量对应的全局R^2\n'
    # 遍历每一种带宽
    for i in range(len(list_b_n)):
        # if list_b_n[i] == 30:
        #     print('\nb_n: {}'.format(list_b_n[i]))
        square_sum = 0
        list_y = []
        for j in range(number_random, NUMBER):
            # matrix_w = cal_weight_matrix(source_data['x'][j], source_data['y'][j], source_data['x'], source_data['y'],
            #                              list_b[i], 'fixed', 'bi-square', number_random)
            matrix_w = cal_weight_matrix(source_data['x'][index_shuffle[j]], source_data['y'][index_shuffle[j]],
                                         source_data['time'][index_shuffle[j]], source_data_random['x'],
                                         source_data_random['y'], source_data_random['time'],
                                         list_b_n[i], 'adaptive', 'bi-square', number_random)
            # print('权重矩阵 W: \n{}'.format(matrix_w))
            test_matrix_b = cal_result(matrix_w, matrix_xt_random, matrix_x_random, matrix_y_random)
            # print('回归系数矩阵 b: \n{}'.format(test_matrix_b))
            matrix_x_to_predict = np.mat([1, source_data['aod'][index_shuffle[j]], source_data['t'][index_shuffle[j]],
                                          source_data['p'][index_shuffle[j]], source_data['ws'][index_shuffle[j]],
                                          source_data['rh'][index_shuffle[j]], source_data['dem'][index_shuffle[j]],
                                          source_data['ndvi'][index_shuffle[j]]])
            y_pre = cal_predict(test_matrix_b, matrix_x_to_predict)
            square_sum += (y_pre - source_data['pm2_5'][index_shuffle[j]]) ** 2
            # if list_b_n[i] == 30:
            #     print(source_data['pm2_5'][index_shuffle[j]], y_pre)
            list_y.append(y_pre)
        # if list_b_n[i] == b_n_final:
        #     list_train_y_predict = list_y.copy()
        aic = math.log(square_sum / number_random) + 2 * (k + 1) / number_random
        list_aic.append(aic)
        # print('b: {}   r^2: {} {}'.format(list_b[i], test_global_r(list_y), 1-(square_sum / y_s2)))
        # print('b_n: {}   r^2: {} {}'.format(list_b_n[i], test_global_r(list_y), 1 - (square_sum / y_s2)))
        text_str += 'b_n: {}   r^2: {:.4f} {:.4f}\n'.format(list_b_n[i], test_global_r(list_y), 1 - (square_sum / y_s2))
    return list_aic


def cal_predict(matrix_b, x_to_predict):
    y_predict = np.dot(x_to_predict, matrix_b)
    return y_predict[0, 0]


def test_global_r(list_y):
    res1 = 0
    global y_s2
    global y_avg
    for i in list_y:
        res1 += (i - y_avg)**2
    return res1 / y_s2


def test_local_r(matrix_w_r, list_train_y_predict_local):
    global NUMBER, source_data
    count, sum1 = 0, 0
    for i in range(NUMBER):
        if matrix_w_r[i, i] > 0:
            count += 1
            sum1 += source_data['pm2_5'][i]
    y_avg_local = sum1 / count
    y_predict_sum, y_real_sum = 0, 0
    for y_predict in list_train_y_predict_local:
        y_predict_sum += (y_predict - y_avg_local) ** 2
    for y_real in source_data['pm2_5']:
        y_real_sum += (y_real - y_avg_local) ** 2
    # print('local R^2: {}'.format(y_predict_sum / y_real_sum))
    return y_predict_sum / y_real_sum


def get_cor(x_index, y_index):
    global x_min, x_max, y_min, y_max
    cor = dict()
    cor.clear()
    cor['x'] = x_min + 250 + y_index*500
    cor['y'] = y_max - 250 - x_index*500
    return cor


def gtwr_predict(processor_index, num_processor, matrix_xt_gwr, matrix_x_gwr, matrix_y_gwr, list_train_y_predict_local,
                 date, utc):
    global source_data, img_intercept, img_aod, img_t, img_p, img_ws, img_rh, img_dem, img_ndvi, img_local_r
    global b_n_final, b_final, NUMBER, line_num, row_num
    result = dict()
    start = int(processor_index * (line_num // num_processor))
    end = int((processor_index + 1) * (line_num // num_processor))
    if processor_index == num_processor - 1:
        end = line_num
    start_time1 = time.process_time()
    print('进程 {} 部署完成: 行号范围: {} ~ {}'.format(processor_index, processor_index, start, end))
    for i in range(start, end):
        start_time2 = time.process_time()
        for j in range(row_num):
            cor = get_cor(i, j)
            # matrix_w = cal_weight_matrix(cor['x'], cor['y'], source_data['x'], source_data['y'],
            #                              b_final, 'fixed', 'bi-square', NUMBER)
            matrix_w = cal_weight_matrix(cor['x'], cor['y'], date + utc/24, source_data['x'], source_data['y'],
                                         source_data['time'], b_n_final, 'adaptive', 'bi-square', NUMBER)
            matrix_b = cal_result(matrix_w, matrix_xt_gwr, matrix_x_gwr, matrix_y_gwr)
            img_intercept[i][j] = matrix_b[0, 0]
            img_aod[i][j] = matrix_b[1, 0]
            img_t[i][j] = matrix_b[2, 0]
            img_p[i][j] = matrix_b[3, 0]
            img_ws[i][j] = matrix_b[4, 0]
            img_rh[i][j] = matrix_b[5, 0]
            img_dem[i][j] = matrix_b[6, 0]
            img_ndvi[i][j] = matrix_b[7, 0]
            img_local_r[i][j] = test_local_r(matrix_w, list_train_y_predict_local)
        end_time = time.process_time()
        print('进程 {}: 第{:5d}行计算完成! 本行用时: {:6.3f}s, '
              '本进程累计用时: {:8.3f}s.'.format(processor_index, i+1, end_time - start_time2, end_time - start_time1))
    result['intercept'] = img_intercept
    result['aod'] = img_aod
    result['t'] = img_t
    result['p'] = img_p
    result['ws'] = img_ws
    result['rh'] = img_rh
    result['dem'] = img_dem
    result['ndvi'] = img_ndvi
    result['local_r'] = img_local_r
    print('进程 {} 计算完成!'.format(processor_index))
    return result


def dispose(matrix_xt_dis, matrix_x_dis, matrix_y_dis):
    global list_train_y_predict, start_date, end_date, hour_interval
    global img_intercept, img_aod, img_t, img_p, img_ws, img_rh, img_dem, img_ndvi, proj_templete, geotrans_templete
    global line_num, row_num, img_local_r

    for date in range(start_date, end_date + 1):
        for utc in range(0, 24, hour_interval):
            results = []
            print('即将开始下一轮预测，预测时间: 2019年3月{:2d}日{:2d}时整(UTC)'.format(date, utc))
            print('多进程计算开始部署! 请稍候...')
            num_pro = int(0.9 * mp.cpu_count())
            pool = mp.Pool()
            for _iter in range(num_pro):
                results.append(pool.apply_async(gtwr_predict, args=(_iter, num_pro, matrix_xt_dis, matrix_x_dis,
                                                                    matrix_y_dis, list_train_y_predict, date, utc)))
            pool.close()
            pool.join()
            print('多进程计算完成!')
            # gwr_predict(0, 1, matrix_xt_dis, matrix_x_dis, matrix_y_dis)
            for processor_index in range(num_pro):
                start = int(processor_index * (line_num // num_pro))
                end = int((processor_index + 1) * (line_num // num_pro))
                if processor_index == num_pro - 1:
                    end = line_num
                for i in range(start, end):
                    for j in range(row_num):
                        img_intercept[i][j] = results[processor_index].get()['intercept'][i, j]
                        img_aod[i][j] = results[processor_index].get()['aod'][i, j]
                        img_t[i][j] = results[processor_index].get()['t'][i, j]
                        img_p[i][j] = results[processor_index].get()['p'][i, j]
                        img_ws[i][j] = results[processor_index].get()['ws'][i, j]
                        img_rh[i][j] = results[processor_index].get()['rh'][i, j]
                        img_dem[i][j] = results[processor_index].get()['dem'][i, j]
                        img_ndvi[i][j] = results[processor_index].get()['ndvi'][i, j]
                        img_local_r[i][j] = results[processor_index].get()['local_r'][i, j]
            Grid.write_img('./result/c_intercept-' + str(date) + '-' + str(utc) + '.tif', img_intercept, proj_templete,
                           geotrans_templete, 'tif')
            Grid.write_img('./result/c_aod-' + str(date) + '-' + str(utc) + '.tif', img_aod, proj_templete,
                           geotrans_templete, 'tif')
            Grid.write_img('./result/c_t-' + str(date) + '-' + str(utc) + '.tif', img_t, proj_templete,
                           geotrans_templete, 'tif')
            Grid.write_img('./result/c_p-' + str(date) + '-' + str(utc) + '.tif', img_p, proj_templete,
                           geotrans_templete, 'tif')
            Grid.write_img('./result/c_ws-' + str(date) + '-' + str(utc) + '.tif', img_ws, proj_templete,
                           geotrans_templete, 'tif')
            Grid.write_img('./result/c_rh-' + str(date) + '-' + str(utc) + '.tif', img_rh, proj_templete,
                           geotrans_templete, 'tif')
            Grid.write_img('./result/c_dem-' + str(date) + '-' + str(utc) + '.tif', img_dem, proj_templete,
                           geotrans_templete, 'tif')
            Grid.write_img('./result/c_ndvi-' + str(date) + '-' + str(utc) + '.tif', img_ndvi, proj_templete,
                           geotrans_templete, 'tif')
            Grid.write_img('./result/local_r-' + str(date) + '-' + str(utc) + '.tif', img_local_r, proj_templete,
                           geotrans_templete, 'tif')
            print('本次预测回归系数栅格输出完成!')
            print('本次PM2.5 预测开始! 请稍候...')
            data_prepared = dict()
            data_prepared['c_intercept'] = img_intercept
            data_prepared['c_aod'] = img_aod
            data_prepared['c_t'] = img_t
            data_prepared['c_p'] = img_p
            data_prepared['c_ws'] = img_ws
            data_prepared['c_rh'] = img_rh
            data_prepared['c_dem'] = img_dem
            data_prepared['c_ndvi'] = img_ndvi
            data_prepared['proj'] = proj_templete
            data_prepared['geo'] = geotrans_templete
            grid_calculation(data_prepared, date, utc)
            print('本轮预测完成!')


def grid_calculation(data, date, utc):
    dic1 = './img_data/'
    dic2 = '-' + str(date) + '-' + str(utc) + '.tif'
    cs = ['aod', 't', 'p', 'ws', 'rh', 'dem', 'ndvi']
    for index in range(len(cs)):
        if index < len(cs) - 2:
            dic = dic1 + cs[index] + dic2
        else:
            dic = dic1 + cs[index] + '.tif'
        img_temp, proj_temp, geotrans_temp = Grid.read_img(dic)
        data[cs[index]] = img_temp
    img_result = data['c_intercept'] + data['c_aod'] * data['aod'] + data['c_t'] * data['t'] + \
                 data['c_p'] * data['p'] + data['c_ws'] * data['ws'] + data['c_rh'] * data['rh'] + \
                 data['c_dem'] * data['dem'] + data['c_ndvi'] * data['ndvi']
    Grid.write_img('./result/pm25_predict' + dic2,
                   img_result, data['proj'], data['geo'], 'tif')
    print('本次PM2.5 预测完成! 栅格已输出!')


if __name__ == '__main__':
    # 数据输入、准备阶段
    print('全部的训练样本数目: {}   全部的解释变量个数: {}'.format(NUMBER, k))
    text_str += 'GWR 结果\n'
    text_str += '全部的训练样本数目: {}   全部的解释变量个数: {}\n'.format(NUMBER, k)
    y_avg = sum(source_data['pm2_5']) / NUMBER  # 训练样本被解释变量平均值
    for pm25 in source_data['pm2_5']:
        y_s2 += (pm25 - y_avg)**2  # 训练样本被解释变量方差
    print('y_avg: {}   y_s2: {}'.format(y_avg, y_s2))
    matrix_xt = np.mat([[1]*NUMBER, source_data['aod'], source_data['t'], source_data['p'], source_data['ws'],
                        source_data['rh'], source_data['dem'], source_data['ndvi']])  # 解释变量矩阵
    matrix_x = matrix_xt.T  # 解释变量矩阵转置
    matrix_yt = np.mat(source_data['pm2_5'])  # 被解释变量矩阵转置
    matrix_y = matrix_yt.T  # 被解释变量矩阵
    # print('解释变量矩阵 X: \n{}'.format(matrix_x))
    # print('被解释变量矩阵 Y: \n{}'.format(matrix_y))

    # 确定最佳带宽
    print('赤池信息准则检验开始! 请稍候...')
    global_aicc_list = aic_test(matrix_xt, matrix_x, matrix_y)
    aicc_list = aic_test_random()
    fig = plt.figure(figsize=(10, 6))
    chart = np.arange(1, int(0.69*NUMBER))
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.plot(chart, aicc_list, c='red')
    plt.title('AICc与带宽大小b_n之间的走势')
    plt.xlabel('第x个带宽大小')
    plt.ylabel('AICc')
    # plt.savefig('./result/aic.png')
    plt.show()
    print('赤池信息准则检验完成!')

    # 输出模型拟合信息
    print('模型拟合全局信息输出开始! 请稍候...')
    text_str += '\n不同的带宽数量对应的AICc\n'
    for i in range(1, int(0.69*NUMBER)):
        text_str += 'b_n: {}   AICc: {:.4f}\n'.format(i, aicc_list[i-5])
    text_str += '\n选定的带宽数量为 {}\n'.format(b_n_final)
    text_file = open(r'.\result\final_result.txt', 'w')
    text_file.write(text_str)
    text_file.close()
    print('模型拟合全局信息输出完成!')

    # 生成回归系数栅格图、预测值栅格图
    print('模型拟合开始! 请稍候...')
    dispose(matrix_xt, matrix_x, matrix_y)
    print('模型拟合完成!')
    print('全部任务完成!')
