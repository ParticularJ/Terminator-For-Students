import os
import argparse
import pickle
from collections import defaultdict
from utils import *
import numpy as np
import math
import cv2

FLAGS = None
IMAGE_SIZE = (6553, 3456)


def parser():
    args = argparse.ArgumentParser()
    args.add_argument('--images_path', default='/home/enningxie/Documents/DataSets/face_images')
    args.add_argument('--pickle_path', default='../pickle_path')
    args.add_argument('--faces_path', default='/home/enningxie/Documents/DataSets/face_rec/faces_path')
    return args.parse_args()


def restore_op(pickle_path):
    line_path = os.path.join(pickle_path, 'line.pkl')
    line_02_path = os.path.join(pickle_path, 'line_02.pkl')
    locations_path = os.path.join(pickle_path, 'all_locations.pkl')
    with open(line_02_path, 'rb') as f:
        line_02_data = pickle.load(f)
    with open(line_path, 'rb') as f:
        line_data = pickle.load(f)
    with open(locations_path, 'rb') as f:
        all_locations = pickle.load(f)
    return line_data, line_02_data, all_locations


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


def save_faces_to_disk(image, faces_path, file_name, group, count):
    for key in group.keys():
        for key_ in group[key].keys():
            sub_folder_name = key + '_' + key_
            file_folder_path = os.path.join(faces_path, sub_folder_name)
            if not os.path.exists(file_folder_path):
                os.mkdir(file_folder_path)
            file_path = os.path.join(file_folder_path, file_name)
            top, right, bottom, left = group[key][key_]
            if top <= 30 or right <= 30 or bottom <= 30 or left <= 30:
                cv2.imwrite(file_path, image[top: bottom, left: right])
            else:
                cv2.imwrite(file_path, image[top - 30: bottom + 30, left - 30: right + 30])
    print('processing {0}.'.format(count))


if __name__ == '__main__':
    FLAGS = parser()
    # restore_op
    line_data, line_02_data, all_locations = restore_op(FLAGS.pickle_path)
    # process_images_op
    name_list = [i * 100 for i in range(1, 133)]
    count = 0
    for name, locations in zip(name_list, all_locations):
        image_name = str(name)+'.jpg'
        image_path = os.path.join(FLAGS.images_path, image_name)
        image_data = cv2.imread(image_path)
        image_resized = image_resize(image_data, IMAGE_SIZE)
        group = sort_locations(locations, line_data, line_02_data)
        save_faces_to_disk(image_resized, FLAGS.faces_path, image_name, group, count)
        count += 1




