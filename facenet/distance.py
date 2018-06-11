import numpy as np
import pickle
import math
from collections import defaultdict
import cv2
from utils import *


pkl_path = '/home/enningxie/Documents/DataSets/face_rec/pkl_images/10600.pkl'
image_path = '/home/enningxie/Documents/DataSets/face_images/10600.jpg'

group = defaultdict(dict)

with open(pkl_path, 'rb') as f:
    locations = pickle.load(f)
with open('./line_02.pkl', 'rb') as f:
    line_02_data = pickle.load(f)
with open('./line.pkl', 'rb') as f:
    line_data = pickle.load(f)

for location in locations:
    top, right, bottom, left = location
    x = (right + left) / 2
    y = (top + bottom) / 2
    distance_row = []
    distance_col = []
    for k, b in line_02_data:
        dis = abs(k * x + b - y) / math.sqrt(1 + math.pow(k, 2))
        distance_row.append(dis)
    dis_row_num = np.asarray(distance_row)
    group_num_row = np.argmin(dis_row_num)

    for k, b in line_data:
        dis = abs(k * x + b - y) / math.sqrt(1 + math.pow(k, 2))
        distance_col.append(dis)
    dis_col_num = np.asarray(distance_col)
    group_num_col = np.argmin(dis_col_num)
    group[str(group_num_row)][str(group_num_col)] = location

# for key in group.keys():
#     group[key] = sorted(group[key], key=lambda x: x[1])
#     print('{0}: {1}.'.format(key, group[key]))
count = 0
for key in group.keys():
    for key_ in group[key].keys():
        print(group[key][key_])
        print(key + '_' + key_)
        count += 1

image_data = cv2.imread(image_path)
image_reshape = image_resize(image_data, (6553, 3456))

image_data = draw_op_dict(image_reshape, group)

cv2.imwrite('./test_cv2_05.jpg', image_data)



# distance = []
# for k, b in data:
#     dis = abs(k*point[0]+b-point[1])/math.sqrt(1+math.pow(k, 2))
#     distance.append(dis)
#
# dis_num = np.asarray(distance)
#
# print(np.argmin(dis_num))