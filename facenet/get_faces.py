import os
import argparse
import pickle
from collections import defaultdict
from utils import *
import numpy as np
import math

FLAGS = None
IMAGE_SIZE = (6553, 3456)


def parser():
    args = argparse.ArgumentParser()
    args.add_argument('--images_path', default='')
    args.add_argument('--pickle_path', default='')
    return args.parse_args()


def restore_op():
    with open('./line_02.pkl', 'rb') as f:
        line_02_data = pickle.load(f)
    with open('./line.pkl', 'rb') as f:
        line_data = pickle.load(f)
    return line_data, line_02_data


def sort_locations(locations, line_data, line_02_data):
    group = defaultdict(dict)
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
    return group


if __name__ == '__main__':
    FLAGS = parser()


