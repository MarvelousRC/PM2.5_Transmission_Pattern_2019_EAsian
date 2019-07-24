#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <iostream>

#include <gdal.h>
#include <gdal_priv.h>

#define SRC_TIF "F:/2019_shortterm/AOD_to_interpolate/aod-16-7.tif"
#define DST_TIF "F:/2019_shortterm/AOD_final/aod-16-7.tif"

void init_gdal() {
	// 如果需要用到更高级GDAL的特性, 自行查阅文档
	GDALAllRegister();
	CPLSetConfigOption("GDAL_FILENAME_IS_UTF8", "NO");
	CPLSetConfigOption("SHAPE_ENCODING", "");
}

int main(int argc, char **argv) {
	init_gdal();
	
	// 将原始影像文件的第一波段单独输出成新的影像
	// 推荐重点了解一下 RasterIO 这个方法

	// 打开影像文件
	GDALDataset *pSrcDataset = (GDALDataset *)GDALOpen(SRC_TIF, GA_ReadOnly);
	assert(NULL != pSrcDataset);

	// 获取影像的列数和行数
	int nCols = pSrcDataset->GetRasterXSize();
	int nRows = pSrcDataset->GetRasterYSize();
	double *pData = new double[nCols * nRows];
	assert(NULL != pData);

	// cout << 'col: ' << nCols << 'row: ' << nRows;

	// 获取第一个波段(从1开始计)
	GDALRasterBand *pSrcBand = pSrcDataset->GetRasterBand(1);
	assert(NULL != pSrcBand);

	// 读取数据
	CPLErr err = pSrcBand->RasterIO(GF_Read, 0, 0, nCols, nRows, pData, nCols, nRows, GDT_Float64, 0, 0, 0);
	assert(CE_None == err);
	
	double *pOutData = new double[nCols * nRows];
	memcpy(pOutData, pData, nCols * nRows * sizeof(double));
	//std::copy(std::begin(pData), std::end(pData), std::begin(pOutData));

	long int cnt = 0;
	double _eps = 0.0;

	clock_t _start = clock();
	for(int y = 0; y < nRows; y++)
	{
		if(y % 100 == 0) std::cout << y << " " << clock()/1000.0 << std::endl;
		for(int x = 0; x < nCols; x++)
		{
			if(pData[y*nCols+x] <= _eps)
			{
				cnt ++;
				int e = 1;
				while(true)
				{
					int flag = 0;
					for(int t = x-e; t <= x+e; t++)
					{
						if( 0 <= t && t < nCols)
						{
							if((y - e >= 0 && pData[(y-e)*nCols + (t)] > _eps) || (y + e < nRows && pData[(y+e)*nCols + (t)] > _eps) )
							{
								flag = 1;
								break;
							}
						}
					}
					for(int t = y-e; t <= y+e; t++)
					{
						if( 0 <= t && t < nRows)
						{
							if((x - e >= 0 && pData[(t)*nCols + (x-e)] > _eps) || (x + e < nCols && pData[(t)*nCols + (x+e)] > _eps) )
							{
								flag = 1;
								break;
							}
						}
					}
					if(flag == 1) break;
					e++;
				}

				double sum1 = 0.0, sum2 = 0.0;
				int ee = e, interval;
				if(e > 150) interval = 30;
				else interval = 2 * ee;

				for(int e = ee; e <= ee + interval; e+=1)
				{
					for(int t = x-e; t <= x+e; t++)
					{
						if( 0 <= t && t < nCols)
						{
							if(y - e >= 0 && pData[(y-e)*nCols + (t)] > _eps)
							{
								sum1 += pData[(y-e)*nCols + (t)] / (e*e + (t-x)*(t-x));
								sum2 += 1.0 / (e*e + (t-x)*(t-x));
							}
							if(y + e < nRows && pData[(y+e)*nCols + (t)] > _eps)
							{
								sum1 += pData[(y+e)*nCols + (t)] / (e*e + (t-x)*(t-x));
								sum2 += 1.0 / (e*e + (t-x)*(t-x));
							}
						}
					}
					for(int t = y-e+1; t < y+e; t++)
					{
						if( 0 <= t && t < nRows)
						{
							if(x - e >= 0 && pData[(t)*nCols + (x-e)] > _eps)
							{
								sum1 += pData[(t)*nCols + (x-e)] / (e*e + (t-y)*(t-y));
								sum2 += 1.0 / (e*e + (t-y)*(t-y));
							}
							if(x + e < nCols && pData[(t)*nCols + (x+e)] > _eps)
							{
								sum1 += pData[(t)*nCols + (x+e)] / (e*e + (t-y)*(t-y));
								sum2 += 1.0 / (e*e + (t-y)*(t-y));
							}
						}
					}
				}
				pOutData[y*nCols + x] = sum1 / sum2;
			}
		}
	}
	clock_t _end = clock();
	std::cout << (_end - _start) / 1000.0 << std::endl;

	std::cout << cnt << '/' << nCols*nRows << std::endl;


	// 获取投影信息
	const char* szProjectionRef = pSrcDataset->GetProjectionRef();
	assert(NULL != szProjectionRef);
	double dGeoTransform[6];
	err = pSrcDataset->GetGeoTransform(dGeoTransform);
	assert(CE_None == err);

	// 创建新的影像
	GDALDriver *pDriver = GetGDALDriverManager()->GetDriverByName("GTIFF");
	assert(NULL != pDriver);
	GDALDataset *pDstDataset = pDriver->Create(DST_TIF, nCols,nRows, 1, GDT_Float64, 0);
	assert(NULL != pDstDataset);

	// 设置投影信息
	pDstDataset->SetProjection(szProjectionRef);
	pDstDataset->SetGeoTransform(dGeoTransform);

	// 将数据写入新影像
	GDALRasterBand *pDstBand = pDstDataset->GetRasterBand(1);
	assert(NULL != pDstBand);
	err = pDstBand->RasterIO(GF_Write, 0, 0, nCols, nRows, pOutData, nCols, nRows, GDT_Float64, 0, 0, 0);
	assert(CE_None == err);
	
	// 关闭影像文件
	delete[] pData; 
	delete[] pOutData;
	GDALClose(pSrcDataset);
	GDALClose(pDstDataset);

	system("pause");

	return 0;
}
