# 气溶胶光学厚度(AOD)反演

## 什么是AOD？

首先我们需要明白我们的研究对象的定义，描述一个物质可以直接从它的组成上来描述：**云、雾、烟尘**，这些所有都算是，pm2.5只是其中很小的一部分。
NASA有提供直接的AOD产品——MYD04(550nm)。AOD除了和大气的组成直接相关外，还和光的波长有关。

## 反演的核心

地表反射 + 大气反射 = 表观反射，也就是很多人说的地气解耦。

![](http://latex.codecogs.com/gif.latex?\rho_{TOA}(\theta_{S},\theta_{v})=\rho_{0}(\theta_{S},\theta_{v})+\frac{T(\theta_{S})T(\theta_{v})\rho(\theta_{S},\theta_{v})}{1-S\cdot{\rho_{S}}(\theta_{S},\theta_{v})})

![](http://latex.codecogs.com/gif.latex?\rho_{TOA})是表观反射率，![](http://latex.codecogs.com/gif.latex?\rho_{s})是地表反射率，其余的是与大气相关参数。
其中地表反射率还有一个公式：

![](http://latex.codecogs.com/gif.latex?\rho_{TOA}=\frac{\pi{L_{\lambda}}D^2}{ESUN_{\lambda}cos\theta})

其中D是天文单位的日地距离，恰巧日地距离就是一个天文单位。 ![](http://latex.codecogs.com/gif.latex?L_\lambda)是经过辐射校正之后的辐亮度，任何卫星应该在拍摄的时候记录下来gain和bias的数值，就是一个线性的变换。![](http://latex.codecogs.com/gif.latex?\theta)是太阳天顶角，也是卫星元数据的一部分。![](http://latex.codecogs.com/gif.latex?ESUN_{\\lambda})是大气顶部的太阳辐照度值，是波长的函数，有论文指出了我们需要用到的波段的该值。通过此公式，我们即可由遥感原始数据的DN值推导出表观反射率。

> At many wavelengths of visible light, the contrast between aerosols and the surface is difficult to discern, but in the 412 nm band——the "deep blue" band, aerosol signals tend to be bright and surface features dark.    ——NASA

根据以上事实，我们的表观反射率的数据选择为GOCI的412nm（第一波段），在此波段下的分子散射和吸收都远远的低于其它波段，故气溶胶垂直廓线的影响对深蓝波段气溶胶光学厚度反演的影响甚微。在这里我们运用的地表反射率产品直接是MYD09。其次，当地表反射率较小时，卫星观测到的表观反射率与气溶胶光学厚度有着很好的线性相关性。且蓝波段的敏感度很大：蓝波段地表反射率的分布比较集中，80%介于0～0.1之间，60%小于0.05，只有10%高于0.2。对于MYD09的数据，我们只需剔除那些高于0.1的数值。

在大气科学领域还有一个6S模型用于描述大气参数与AOD数值的关系，所以最终我们是通过查表获得最贴切的AOD数值。

## Command:

> Example:

```shell
python aod_retrieval_db.py --goci ../aod_retrieval_data/L1B1_land/L1B1_land-14-1.tif --solz ../aod_retrieval_data/SOLZ_land/SOLZ_land-14-1.tif --myd09 ../aod_retrieval_data/MYD09/MYD09A1-project1.tif --cloud ../aod_retrieval_data/cloud_land/cloud_land-14-1.tif --lut ../aod_retrieval_data/LUT/modis_lut_m.txt --output ../aod_retrieval_res/
```

## 用什么测评

NASA有相关的监测站是专门用来测AOD，以及他们有MYD04的产品直接是AOD的栅格，可以用作检验反演算法的有效性。
