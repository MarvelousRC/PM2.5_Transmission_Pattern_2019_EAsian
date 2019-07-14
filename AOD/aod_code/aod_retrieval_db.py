"""
AOD retrieval algorithm: Deep Blue
==================================
input data:
---
1 GOCI::band1(412nm)
2 SOLZ::solar zenith angle
3 MODIS09::band3
4 cloud_mask
5 LUT::6S model

output:
---
each hour's AOD
"""


from osgeo import gdal
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import sys
import time
import datetime
import multiprocessing


__author__ = "Andrian Lee"
__date__ = "2019/07/14"


class Grid(object):
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

    def write_img(self, _file, img_data, img_proj, img_geotrans, _format):
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


def LUT_read(_file):
    f = open(_file)
    lines = f.readlines()
    f.close()
    res = []
    for line in lines:
        line = [float(x) for idx, x in enumerate(line.strip().split()) if idx in [0, 1, 2, 6]]
        res.append(line)
    return res


def AOD_Deepblue(data, _file, mod09, cloud):
    lut = LUT_read(_file)
    row, col = data.shape[0], data.shape[1]
    tot = row * col

    # Choose the qualified deep_blue area
    data, cnt_np = choose_db_cloud(data, mod09, cloud)
    print("To be processed", tot - cnt_np, "/", tot)

    print('DB start time', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    pool = multiprocessing.Pool()
    cpus = multiprocessing.cpu_count()
    num_process = int(cpus * 0.9)
    results = []
    for _iter in range(num_process):
        results.append(pool.apply_async(get_aod, args=(_iter, row, num_process, data, lut, mod09)))
    pool.close()
    pool.join()

    aod_dict = {}
    for result in results:
        aod_dict.update(result.get())

    out_aod = []
    for k in sorted(aod_dict):
        out_aod.append(aod_dict[k])
    out_aod = np.array(out_aod)

    return out_aod


def get_aod(start_pos, end_pos, step_size, data, lut, mod09):
    """Sub-process for parallel processing on AOD retrieval"""
    out = {}

    print('Process'+str(start_pos), 'started...')
    for idi in range(start_pos, end_pos, step_size):  # load balancing
        aod = np.zeros(data.shape[1])  # col

        _start = datetime.datetime.now()
        for idj, j in enumerate(data[idi]):
            if j != 0.0 and j != -1.0:  # ignore mask and nodata
                aod_v = LUT_match(data, lut, idi, idj, mod09)  # 0.02s for 0.1 step-wise 6s-LUT each pixel
                aod[idj] = aod_v
        _end = datetime.datetime.now()
        print('Process'+str(start_pos), str(idi)+'-th row finished, time cost:', _end - _start, time.ctime())

        out.update({idi: aod})
    print('Process'+str(start_pos), 'ended...')

    return out


def choose_db_cloud(data, mod09, cloud):
    """
        nodata -> -1.0
        db, cloud(mask) -> 0.0
    """
    nodata = np.min(data)
    nodata_index = data == nodata
    data[nodata_index] = -1.0

    mask_db = mod09 > 0.1
    data[mask_db] = 0.0

    mask_cloud = cloud == 1
    data[mask_cloud] = 0.0

    num_nodata = np.count_nonzero(nodata_index)
    num_mask = np.count_nonzero(mask_db)
    num_cloud = np.count_nonzero(mask_cloud)

    return data, num_nodata+num_mask+num_cloud


def LUT_match(data, lut, idi, idj, mod09):
    """look up table"""
    p_aot1 = data[idi][idj]   # band 3
    p_sur = mod09[idi][idj]      # mod09

    min_d = 100
    aod_v = 0
    for i in lut:
        p_aot2 = i[2] + (i[1] * p_sur) / (1 - i[0] * p_sur)
        _delta = abs(p_aot2 - p_aot1)
        if _delta < min_d:
            aod_v = i[3]
            min_d = _delta
    return aod_v


def Fast_interpolate(aod, data):
    """Interpolate the mask area -> 0.0"""
    row, col = aod.shape[0], aod.shape[1]
    for idi, i in enumerate(aod):
        # print('interpolate', idi)
        for idj, j in enumerate(i):
            if data[idi][idj] == 0.0:
                # assign range
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

                # calculate mean
                cnt = 0
                tot = 0
                for t in range(idj-x, idj+x+1):
                    if idi - x >= 0 and idi + x < row and 0 <= t < col:
                        if aod[idi + x][t] > 0.0:
                            tot += aod[idi + x][t]
                            cnt += 1
                        if aod[idi - x][t] > 0.0:
                            tot += aod[idi - x][t]
                            cnt += 1
                for t in range(idi-x+1, idi+x):
                    if 0 <= t < row and idj - x >= 0 and idj + x < col:
                        if aod[t][idj - x] > 0.0:
                            tot += aod[t][idj - x]
                            cnt += 1
                        if aod[t][idj + x] > 0.0:
                            tot += aod[t][idj + x]
                            cnt += 1
                if cnt == 0:
                    print("strange:", idi, " ", idj)
                    print("x", x)
                else:
                    aod[idi][idj] = tot / cnt
    return aod


def preprocess_GOCI(data, solz):
    """ DN -> apparent reflectance"""

    _ESUN = 1796.65  # GOCI::band_421nm
    _D = 1
    gain = 1e-6
    bias = 0

    data = (data * gain + bias) * (_D * _D) * np.pi / _ESUN
    nodata = np.min(data)
    tmp = np.where(data != nodata)  # index
    data[tmp] / np.cos(solz[tmp]*np.pi/180)

    return data


def main():
    parser = ArgumentParser("aod_retrieval", formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler='resolve')
    parser.add_argument('--goci')
    parser.add_argument('--solz')
    parser.add_argument('--myd09')
    parser.add_argument('--cloud')
    parser.add_argument('--lut')
    parser.add_argument('--output')
    args = parser.parse_args()

    run = Grid()

    raster_toa = args.goci
    data, proj, geotrans = run.read_img(raster_toa)

    raster_solz = args.solz
    solz, proj_solz, geotrans_solz = run.read_img(raster_solz)

    data = preprocess_GOCI(data, solz)

    raster_surface = args.myd09  # nodata: -INFINITE, max:0.77
    mod09, proj1, geotrans1 = run.read_img(raster_surface)

    raster_cloud = args.cloud
    cloud, proj_cloud, geotrans_cloud = run.read_img(raster_cloud)

    lut_filepath = args.lut
    # multi-process here
    out_data0 = AOD_Deepblue(data, lut_filepath, mod09, cloud)
    out_name = ['aod']
    out_name += raster_toa.split('/')[-1][:-4].split('-')[1:]
    out_name = '-'.join(out_name)
    out_file = args.output + out_name + '_tmp.tif'
    plt.imshow(out_data0)
    plt.savefig(args.output + out_name + '_tmp_thumb.jpg')

    run.write_img(out_file, out_data0, proj, geotrans, 'tif')
    print('AOD retrieval finished!!!!')

    _start = datetime.datetime.now()
    out_data = Fast_interpolate(out_data0, data)
    _end = datetime.datetime.now()
    print('Interpolate time cost:', _end - _start)
    plt.imshow(out_data)
    plt.savefig(args.output + out_name + '_thumb.jpg')

    out_file = args.output + out_name + '.tif'
    run.write_img(out_file, out_data, proj, geotrans, 'tif')

    # Re-sampling
    # shrink_NEAREST = cv2.resize(out_data0, out_size, interpolation=cv2.INTER_NEAREST)
    # shrink_LINEAR = cv2.resize(out_data0, out_size, interpolation=cv2.INTER_LINEAR)
    # shrink_AREA = cv2.resize(out_data0, out_size, interpolation=cv2.INTER_AREA)
    # shrink_CUBIC = cv2.resize(out_data0, out_size, interpolation=cv2.INTER_CUBIC)
    # shrink_LANCZOS4 = cv2.resize(out_data0, out_size, interpolation=cv2.INTER_LANCZOS4)


if __name__ == "__main__":
    sys.exit(main())
