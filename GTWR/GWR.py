import numpy as np
import matplotlib.pyplot as plt
import math

NUMBER = 0
k = 0
num_whole = 0


def read_csv_file():
    source_data = {'id': [], 'lon': [], 'lat': [], 'pm2_5': [], 'aod': [], 'p': [], 't': [], 'ws': [], 'dem': [], 'x': [], 'y': []}
    with open(r'.\data\pm2_5.csv', 'r') as file:
        lines = file.read().splitlines()
        for i in range(len(lines)):
            line_list = lines[i].split(',')
            if i != 0:
                source_data['id'].append(int(line_list[0]))
                source_data['lon'].append(float(line_list[1]))
                source_data['lat'].append(float(line_list[2]))
                source_data['pm2_5'].append(float(line_list[3]))
                source_data['aod'].append(float(line_list[4]))
                source_data['p'].append(float(line_list[5]))
                source_data['t'].append(float(line_list[6]))
                source_data['ws'].append(float(line_list[7]))
                source_data['dem'].append(float(line_list[8]))
                source_data['x'].append(float(line_list[9]))
                source_data['y'].append(float(line_list[10]))
        return source_data


def cal_weight(x, y, U, V, b):
    global NUMBER
    result = np.eye(NUMBER)
    for i in range(NUMBER):
        d = math.sqrt((x-U[i])**2 + (y-V[i])**2)
        if d <= b:
            result[i][i] = math.e**(-(d/b)**2)
    return result


def cal_result(matrix_w, matrix_xt, matrix_x, matrix_y):
    temp = np.dot(matrix_xt, matrix_w)
    temp1 = np.dot(temp, matrix_x)
    temp2 = np.dot(temp, matrix_y)
    matrix_b = np.dot(temp1, temp2)
    return matrix_b


def cia_test(source_data, matrix_xt, matrix_x, matrix_y):
    list_b = [1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000]
    list_cia = []
    global NUMBER, k
    for i in range(len(list_b)):
        square_sum = 0
        for j in range(NUMBER):
            matrix_w = cal_weight(source_data['x'][j], source_data['y'][j], source_data['x'], source_data['y'], list_b[i])
            print('W: {}'.format(matrix_w))
            test_matrix_b = cal_result(matrix_w, matrix_xt, matrix_x, matrix_y)
            y_pre = cal_predict(test_matrix_b, matrix_x[j])
            # print(y_pre, matrix_y[j])
            square_sum += (y_pre - matrix_y[j]) ** 2
        aic = math.log(square_sum / NUMBER) + 2 * (k + 1) / NUMBER
        list_cia.append(aic)
    return list_cia


def cal_predict(matrix_b, x_to_predict):
    y_predict = np.dot(x_to_predict, matrix_b)
    return y_predict[0]


if __name__ == '__main__':
    source_data = read_csv_file()
    NUMBER = len(source_data['dem'])
    k = 5
    print('全部的训练样本数目: {}   全部的解释变量个数: {}'.format(NUMBER, k))
    b = 100000
    num_whole = 0
    grid_x, grid_y = [], []
    matrix_xt = np.array([1] * NUMBER + source_data['aod'] + source_data['p'] + source_data['t'] + source_data['ws'] + source_data['dem']).reshape(6, NUMBER)
    matrix_x = matrix_xt.T
    matrix_y = np.array(source_data['pm2_5']).reshape(NUMBER, 1)
    print('X: {}'.format(matrix_x))
    print('Y: {}'.format(matrix_y))
    aicc = cia_test(source_data, matrix_xt, matrix_x, matrix_y)
    print(aicc)
    fig = plt.figure(figsize=(10, 6))
    chart = np.arange(1, 11)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.plot(chart, aicc, c='red')
    plt.title('AICc与带宽大小b之间的走势')
    plt.xlabel('第x个带宽大小')
    plt.ylabel('AICc')
    plt.show()
    for i in range(num_whole):
        matrix_w = cal_weight(grid_x[i], grid_y[i], [], [], b)
        matrix_b = cal_result(matrix_w, matrix_xt, matrix_x, matrix_y)
