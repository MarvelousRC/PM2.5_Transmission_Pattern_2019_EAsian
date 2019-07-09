# 中韩 PM2.5 传导关系研究
![](https://img.shields.io/badge/build-processing-brightgreen.svg)


> 研究区域：GOCI卫星得到的韩国、中国东北部沿海、黄海等海域
> 研究时间：2019年2月底到3月中旬，短时间内密集采样[初步]

项目组成员：陈玮烨、孙克染、李岸洲、杨清杰
## 数据获取  ![Progress](http://progressed.io/bar/5)
+ GOCI => 波段 => AOD
+ 气象 => T, P, 风速, RH
+ 监测站 => pm2.5
+ NDVI, DEM

## 数据预处理  ![Progress](http://progressed.io/bar/5)
+ 几何校正、辐射校正 => GOCI adj.
+ 空间插值 => T, P, 风速, RH
+ 空间连结 => pm2.5

## 建模  ![Progress](http://progressed.io/bar/5)
+ GWR
+ [ Random Forest ]

## 验证  ![Progress](http://progressed.io/bar/5)
+ 检验回归得到的pm2.5质量
+ 变量显著性

## 分析  ![Progress](http://progressed.io/bar/5)
+ 时间序列下pm2.5的轨迹

