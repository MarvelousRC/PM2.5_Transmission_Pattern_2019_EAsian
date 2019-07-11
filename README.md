# 中韩 PM2.5 传导关系研究

![](https://img.shields.io/badge/build-processing-brightgreen.svg)

> 研究区域：GOCI卫星得到的韩国、中国东北部沿海、黄海等海域
> 
> 研究时间：2019年2月底到3月中旬，短时间内密集采样[初步]

项目组成员：陈玮烨、孙克染、李岸洲、杨清杰
## 数据获取  ![Progress](http://progressed.io/bar/5)  
试验[ 7月11日 ]

+ [ ] 区域确定，出shapefile
+ [ ] 监测站：pm2.5
+ 陆地海洋卫星遥感数据获取
  + [ ] GOCI data
  + [ ] MOD09
+ [ ] 气象数据：T, P, 风速, (RH)
+ [ ] 自然数据：NDVI, DEM
+ [ ] AERONET

## 数据预处理  ![Progress](http://progressed.io/bar/5)
试验[ 7月11日 ]
所有数据范围与投影坐标信息统一：


+ AOD **[ 7.11 ]**
  + [ ] 反演算法DB=> AOD，裁掉海洋，和AERONET进行回归分析和校正
  + [ ] 海洋：通过GOCI的AOT产品结果，与陆地进行拼接
+ 空间全域插值气象数据T, P, 风速, (RH)
  + [ ] ArcGIS/QGIS批处理
+ 空间连接（陆地）
  + [ ] 把所有自变量空间关联到PM2.5观测点

## 建模  ![Progress](http://progressed.io/bar/5)
$C_{pm2.5} = f(AOD, climate, geography)$
* **Dependent Variable**
PM2.5 (地面站点 Monitored on the ground)

* **Independent Variables**
**AOD**, Elevation, Air Temperature, Sea Level Pressure, Wind Direction and Speed, Precipitations, Sky Conditions (bool variables: cloudy, sunny, rain...)

- [ ] 借助工具PM2.5反演的LR和GWR试验模型【7月12日】
- [ ] 基于Python的机器学习的PM2.5反演试验模型预计在7月15日出现雏形

* PM2.5反演模型
  * [ ] 回归（陆地）
    * 方法：Statictical [LR, **GWR**], Machine Learning [Random Forest, SVR…]
      * [mGWR](https://github.com/pysal/mgwr)
      * 对于统计回归模型
        * 数据预处理
        * 线性回归需要分析整体模型的线性显著性（F test）
        * 多重共线性检验
        * 需要对变量进行回归关系的显著性检验（t test） 
      * 对于机器学习回归模型
        * [标准化数据](https://docs.microsoft.com/en-us/azure/machine-learning/studio-module-reference/normalize-data)（有些模型要求这么做，另外就是放在统一尺度能够使算法更加高效）
        * Random Forest [Python](https://towardsdatascience.com/random-forest-in-python-24d0893d51c0)
  * [ ] 回归（海洋）
    * NDVI这项改为primary production，其余不变

## 模型验证与预测  ![Progress](http://progressed.io/bar/5)
+ [ ] 模型比较验证，主要对陆地的结果进行
+ [ ] 全地表覆盖的PM2.5计算。

## 分析说明  ![Progress](http://progressed.io/bar/5)
+ [ ] 时间序列下pm2.5的轨迹
+ 分析中国的PM2.5是否对韩国产生影响
  + [ ] 稍宽时间尺度下的中韩间隔式影响
