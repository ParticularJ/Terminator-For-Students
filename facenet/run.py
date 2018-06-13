import argparse
from utils import convert_video_to_images, image_resize, convert_bgr_to_rgb, draw_op_named, convert_rgb_to_bgr
import os
import cv2
import face_recognition as fr
import pickle
import numpy as np
import math

FLAGS = None
IMAGE_SIZE = (6553, 3456)


def parser():
    args = argparse.ArgumentParser()
    args.add_argument('--video_path', default='/var/Data/xz/face/video_path/ch02_20180308163706.mp4')
    args.add_argument('--images_path', default='/var/Data/xz/face/images_path/ch02_20180308163706')
    args.add_argument('--rec_path', default='/var/Data/xz/face/rec_path/ch02_20180308163706')
    args.add_argument('--model_path', default='/var/Data/xz/face/pickle_path/trained_knn_model.clf')
    args.add_argument('--line_path', default='/var/Data/xz/face/pickle_path/line.pkl')
    args.add_argument('--line_02_path', default='/var/Data/xz/face/pickle_path/line_02.pkl')
    return args.parse_args()


def count_in_pic(images_rgb):
    face_locations_list = []
    count = 0
    for image_rgb in images_rgb:
        print('Locations in {0}.'.format(count))
        face_locations = fr.face_locations(image_rgb, number_of_times_to_upsample=0, model="cnn")
        face_locations_list.append(face_locations)
        count += 1
    return face_locations_list


def _images_rgb(images_path):
    images_list = os.listdir(images_path)
    images_rgb = []
    count = 0
    for image in images_list:
        print('rgb in {0}.'.format(count))
        image_path = os.path.join(images_path, image)
        image_data = cv2.imread(image_path)
        image_resized = image_resize(image_data, IMAGE_SIZE)
        image_rgb = convert_bgr_to_rgb(image_resized)
        images_rgb.append(image_rgb)
        count += 1
    return images_list, images_rgb


def _save_result_op(images_rgb, face_locations_list, images_list, model_path, rec_path, line_path, line_02_path):
    with open(line_02_path, 'rb') as f:
        line_02_data = pickle.load(f)
    with open(line_path, 'rb') as f:
        line_data = pickle.load(f)
    count = 0
    for image_rgb, locations, image_name in zip(images_rgb, face_locations_list, images_list):
        stu_name = []
        faces_data = fr.face_encodings(image_rgb, known_face_locations=locations)
        # for face_data in faces_data:
        #     face_name = predict(face_data, model_path=model_path)
        #     stu_name.append(face_name)
        for face_data, location in zip(faces_data, locations):
            face_name = _predict(face_data, location, line_data, line_02_data, model_path=model_path)
            stu_name.append(face_name)
        print(stu_name)
        image_named = draw_op_named(image_rgb, locations, stu_name)
        image_bgr = convert_rgb_to_bgr(image_named)
        save_path = os.path.join(rec_path, image_name)
        cv2.imwrite(save_path, image_bgr)
        print('processing {0} pic.'.format(count))
        count += 1


def predict(face_data, model_path=None, distance_threshold=0.45):
    with open(model_path, 'rb') as f:
        knn_clf = pickle.load(f)

    # Use the KNN model to find the best matches for the test face
    closest_distances = knn_clf.kneighbors([face_data], n_neighbors=1)
    if closest_distances[0][0][0] <= distance_threshold:
        face_name = knn_clf.predict([face_data])[0]

    else:
        face_name = "unknown"

    return face_name


def _predict(face_data, location, line_data, line_02_data, model_path=None, distance_threshold=0.6):
    with open(model_path, 'rb') as f:
        knn_clf = pickle.load(f)
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
    location_name = str(group_num_row) + '_' + str(group_num_col)
    # Use the KNN model to find the best matches for the test face
    closest_distances = knn_clf.kneighbors([face_data], n_neighbors=1)
    # ### version 2.0 ###
    face_name = knn_clf.predict([face_data])[0]
    if face_name != location_name and closest_distances[0][0][0] > 0.31:
        face_name = "unknown"
    # ### version 1.0 ###
    # if closest_distances[0][0][0] <= distance_threshold:
    #     face_name = knn_clf.predict([face_data])[0]
    #     if face_name != location_name and closest_distances[0][0][0] > 0.31:
    #         face_name = "unknown"
    # else:
    #     face_name = "unknown"
    # print('face_name: {0}'.format(face_name))
    # print('location: {0}_{1}'.format(group_num_row, group_num_col))

    return face_name


if __name__ == '__main__':
    FLAGS = parser()
    video_path = FLAGS.video_path
    images_path = FLAGS.images_path
    rec_path = FLAGS.rec_path
    model_path = FLAGS.model_path
    line_path = FLAGS.line_path
    line_02_path = FLAGS.line_02_path
    convert_video_to_images(video_path, images_path)
    images_list, images_rgb = _images_rgb(images_path)
    face_locations_list = count_in_pic(images_rgb)
    _save_result_op(images_rgb, face_locations_list, images_list, model_path, rec_path, line_path, line_02_path)