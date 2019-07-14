import cv2
from osgeo import gdal
import os
import numpy as np
import matplotlib.pyplot as plt
import time
from argparse import ArgumentParser, FileType, ArgumentDefaultsHelpFormatter
import sys
import aod_retrieval as tool
from sklearn.linear_model import LinearRegression
import math

__author__ = "Andrian Lee"
__date__ = "2019/01/09"


run = tool.Grid()
filename1 = 'batch\MYD021KM.A2011358.0540.061_aod_deepblue.img'
filename2 = 'batch\\2011.img'

data, proj, geotrans = run.read_img(filename1)
data1, proj1, geotrans1 = run.read_img(filename2)

row, col = data.shape[0], data.shape[1]


ours = []
mod04 = []
diff = []

cnt = 0
for i in range(0, row):
    for j in range(0, col):
        if data[i][j] >= 0.0 and data1[i][j] >= 0.0:
            _d = abs(data[i][j] - data1[i][j] / 1000.0)
            ours.append(data[i][j])
            mod04.append(data1[i][j] / 1000)
            diff.append(_d)
            if _d < 0.05:
                # print(i, " ", j)
                cnt += 1

print("-"*40)

lr = LinearRegression()
x = np.array(ours).reshape(-1, 1)
y = np.array(mod04).reshape(-1, 1)
lr.fit(x, y)

b = lr.intercept_[0]
k = lr.coef_[0][0]

_d2 = 0
for i in diff:
    _d2 += i * i
rmse = math.sqrt(_d2 / len(diff))


print("参与测评量：", len(diff))
print("精准预测：", cnt)
print("max: ", max(diff))
print("min: ", min(diff))
print("R^2: ", lr.score(x, y))
print("误差平均:", sum(diff) / len(diff))
print("RMSE:", rmse)

plt.scatter(ours, mod04,  color='black')
plt.plot([0, 0.85], [b, k * 0.85 + b], color='blue', linewidth=3)
plt.savefig(filename2[-4] + '.jpg')
plt.show()
