# 中韩 PM2.5 传导关系研究

![](https://img.shields.io/badge/build-processing-brightgreen.svg)

> 研究区域：GOCI卫星得到的韩国、中国东北部沿海、黄海等海域
> 
> 研究时间：2019年2月底到3月中旬，短时间内密集采样[初步]

项目组成员：陈玮烨 孙克染 李岸洲 杨清杰

## 项目研究范围

##### 空间范围

空间上，本项目研究包含的地理区域如下：

* 陆地部分

  * 中国大陆
    * 浙江北部，江苏全境，安徽东部，山东大部，河北东部，北京天津全境，辽宁南部
  * 朝鲜：全域
  * 韩国：全域
  * 日本：九州岛西部

* 海洋

  * 渤海
  * 黄海
  * 东海北部
  * 日本海西部

* 地理空间范围图示

  ![image-20190726105751061](assets/image-20190726105751061.png)

##### 时间范围

据媒体报道，2019年3月14日-17日在韩国境内的PM2.5浓度有明显的提升，韩国舆论认为该情况和中国大陆的空气污染相关。

故在本研究中，我们选取14日-16日的气象、空气质量等数据，对该时间范围内的中韩空气质量的传导关系进行验证。

## 项目时间表

### 数据获取  ![Progress](http://progressed.io/bar/75)
+ [x] 区域确定，出shapefile [全体]
+ [x] 监测站：pm2.5 [李/杨]
+ 陆地海洋卫星遥感数据获取
  + [x] GOCI data [孙]
  + [x] MOD09 [杨]
+ [x] 气象数据：T, P, 风速, (RH) [陈/杨]
+ [x] 自然数据：NDVI, DEM [杨]
+ [x] AERONET [杨]

### 数据预处理  ![Progress](http://progressed.io/bar/75)

所有数据产品的范围、投影坐标信息、尺度分辨率统一


+ **AOD**
  + [x] 反演算法DB=> AOD，裁掉海洋，和AERONET进行回归分析和校正 [李]
  + [x] 海洋：通过GOCI的AOT产品结果，与陆地进行拼接 [孙/杨]
  + [ ] AOD反演正确和精确性的验证
+ 空间全域插值气象数据T, P, 风速, (RH) 
  + [x] ArcGIS/QGIS批处理 [陈/杨]
+ 空间连接（陆地）
  + [x] 把所有自变量空间关联到PM2.5站点 [李/孙]

### 建模  ![Progress](http://progressed.io/bar/50)
$C_{pm2.5} = f(AOD, climate, geography)$
* **Dependent Variable**
PM2.5 (地面站点 Monitored on the ground)

* **Independent Variables**
**AOD**, Elevation, Air Temperature, Sea Level Pressure, Wind Direction and Speed, Precipitations, Sky Conditions (bool variables: cloudy, sunny, rain...)

- [x] 借助工具PM2.5反演的LR和GWR试验模型

* 基于Python的机器学习的PM2.5反演试验模型
  * 回归（陆地）
    * 方法：Statictical [LR, **GWR**], Machine Learning [Random Forest, SVR…]
      * [ ] Multivariate Linear Regression (OLS) 基于最小二乘法的多元线性回归
      
        * [x] ArcGIS分析
      
        * [ ] Python Implementation (Working)
      
        * [ ] 统计检验和分析
      
          * 线性回归需要分析整体模型的线性显著性（F test）
      
          * 多重共线性检验
          * 需要对变量进行回归关系的显著性检验（t test） 
      
      * [x] GWR模型回归分析
        
        * [x] Python实现
          * 预测参数优化
            * 痴池信息准则
            * 适应性带宽
          * 回归和预测
            * 权重矩阵和数据矩阵
            * 预测分析
        
      * [ ] 对于机器学习回归模型
        - [ ] [标准化数据](https://docs.microsoft.com/en-us/azure/machine-learning/studio-module-reference/normalize-data)（有些模型要求这么做，另外就是放在统一尺度能够使算法更加高效
        - [ ] Random Forest [Python](https://towardsdatascience.com/random-forest-in-python-24d0893d51c0) (正在完成)
    
  * [ ] 回归（海洋）
    
    * 海洋AOD有缺失情况，使用*Fast Interpolation*进行补全
    * 海洋PM2.5是否需要另外的模型还需要讨论

### 模型验证与预测  ![Progress](http://progressed.io/bar/50)
+ [ ] 模型比较验证，主要对陆地的结果进行
+ [ ] 全地表覆盖的PM2.5计算

### 分析说明  ![Progress](http://progressed.io/bar/5)

- [ ] 可视化风场
- [ ] 分析中国的PM2.5是否对韩国产生影响
  * 稍宽时间尺度下的中韩间隔式影响







## 后期任务分工

后期（Aug 1st to Sept 15th) 主要任务包含了如下

* #### AOD验证

  * AOD数据融合和调整（岸洲）
  * 借助已有的AERONET数据对AOD的反演结果进行评估（杨神 孙神）

* #### PM2.5反演模型比较

  * **MLR (OLS) 检验**（F线性检验，t变量检验，置信区间预计）	(陈)
    * 已有Jupyter Notebook代码说明
  * **GWR**的调整和出图 (孙指导，全体成员完成)
    * 如果有需要调整的话
  * 象元级的**机器学习模型**（不考虑地理学第一定律的模型）（陈）
    * [SVR](https://scikit-learn.org/stable/modules/svm.html#svm-regression) (Supporting Vector Regressor), [RF](https://scikit-learn.org/stable/modules/ensemble.html#random-forests) (Random Forests) 和 [MLP](https://scikit-learn.org/stable/modules/neural_networks_supervised.html) (Multilayer Perceptron, Neural Network)（现已完成了拟合，还在写预测的部分）
      * 使用 Sci-Kit Learn 提供的框架
      * 适当引入[交叉验证](https://zhuanlan.zhihu.com/p/24825503)来取得更好的模型
  * 对 OLS, RF, NN, GWR 等 PM2.5 反演模型进行**横向比较**（陈，孙）
  * 讨论**改进空间**（全体成员）

* #### 结果解读（全体开会讨论）

  * 风场**可视化**（杨）
    * 如何结合风场的信息来解读？
  * 说明PM2.5场面数据（判读每一个时刻的PM2.5浓度大小的空间分布情况）
    * 空间自相关性分析
    * 热点分析 (Getis-Ord Gi*) （对栅格数据可行性？）[Link](http://desktop.arcgis.com/zh-cn/arcmap/10.3/tools/spatial-statistics-toolbox/h-how-hot-spot-analysis-getis-ord-gi-spatial-stati.htm)
  * 分析PM2.5变化的趋势
    * 判读
    * 时间序列分析？[Link1](https://blog.csdn.net/qq_41518277/article/details/80288031) [Link2](https://www.datacamp.com/community/tutorials/time-series-analysis-tutorial) [Wikipedia](https://en.wikipedia.org/wiki/Time_series)
      * 目前来看由于时间维度不够长，9个时间点似乎不足以进行时序分析
      * 另外对于栅格数据而言，每一个空间点都会形成一个时间序列，如何建模说明？
      * [ARIMA](https://en.wikipedia.org/wiki/Autoregressive_integrated_moving_average)

* #### 结果呈现

  * 文档
    * 参考文献整理和归档
      * 生成项目参考文献的列表
    * **项目说明报告和结果报告**
      * 参照论文的规格
    * 实验报告
      * 主要是每一个Part的README
  * PPT
    * 从项目说明和结果分析报告中提取并说明