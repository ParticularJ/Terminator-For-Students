import numpy as np
import pickle
import math
from collections import defaultdict
import cv2
from utils import *


pkl_path = '/home/enningxie/Documents/DataSets/face_rec/pkl_images/10600.pkl'
image_path = '/home/enningxie/Documents/DataSets/face_images/10600.jpg'

group = defaultdict(list)

with open(pkl_path, 'rb') as f:
    locations = pickle.load(f)
with open('./line.pkl', 'rb') as f:
    data = pickle.load(f)

for location in locations:
    top, right, bottom, left = location
    x = (right + left) / 2
    y = (top + bottom) / 2
    distance = []
    for num, (k, b) in enumerate(data):
        dis = abs(k*x+b-y)/math.sqrt(1+math.pow(k, 2))
        distance.append(dis)
    dis_num = np.asarray(distance)
    group_num = np.argmin(dis_num)
    group[str(group_num)].append(location)

for key in group.keys():
    group[key] = sorted(group[key])
    print('{0}: {1}.'.format(key, group[key]))




image_data = cv2.imread(image_path)
image_reshape = image_resize(image_data, (6553, 3456))

image_data = draw_op_dict(image_reshape, group)

cv2.imwrite('./test_cv2_03.jpg', image_data)



# distance = []
# for k, b in data:
#     dis = abs(k*point[0]+b-point[1])/math.sqrt(1+math.pow(k, 2))
#     distance.append(dis)
#
# dis_num = np.asarray(distance)
#
# print(np.argmin(dis_num))