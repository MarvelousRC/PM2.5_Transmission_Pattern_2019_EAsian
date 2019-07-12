import numpy as np
import math

def read_csv_file():
    source_data = {'id': [], 'lon': [], 'lat': [], 'pm2_5': [], 'aod': [], 'p': [], 't': [], 'ws': [], 'dem': []}
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
        return source_data

def calWeight(x, y, U, V, NUMBER):
    result = np.eye(NUMBER)
    b = 100000
    for i in range(NUMBER):
        d = math.sqrt((x-U[i])**2 + (y-V[i])**2)
        if d <= b:
            result[i][i] = math.e**(-(d/b)**2)
    return result

if __name__ == '__main__':
    source_data = read_csv_file()
    NUMBER = len(source_data['dem'])
    # print(source_data)
    Xt = np.array([1] * NUMBER + source_data['aod'] + source_data['p'] + source_data['t'] + source_data['ws'] + source_data['dem']).reshape(6, NUMBER)
    X = Xt.T
    Y = np.array(source_data['pm2_5']).reshape(1, NUMBER)
    res = calWeight(0, 0, [], [], NUMBER)
    print(res)
    print(X)
