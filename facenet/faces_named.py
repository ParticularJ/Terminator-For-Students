import argparse
import os
import math
import face_recognition
import pickle
from sklearn import neighbors
from collections import defaultdict
import cv2
import numpy as np
from utils import *

FLAGS = None
IMAGE_SIZE = (6553, 3456)


def parser():
    args = argparse.ArgumentParser()
    args.add_argument('--pickle_path', default='../pickle_path/')
    args.add_argument('--images_path', default='/home/enningxie/Documents/DataSets/face_images')
    args.add_argument('--images_save_path', default='/home/enningxie/Documents/DataSets/face_rec/face_recognition_03')
    return args.parse_args()


def train(X, y, model_save_path=None, n_neighbors=None, knn_algo='ball_tree', verbose=False):
    # Determine how many neighbors to use for weighting in the KNN classifier
    if n_neighbors is None:
        n_neighbors = int(round(math.sqrt(len(X))))
        if verbose:
            print("Chose n_neighbors automatically:", n_neighbors)

    # Create and train the KNN classifier
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    knn_clf.fit(X, y)

    # Save the trained KNN classifier
    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump(knn_clf, f)

    return knn_clf


def predict(face_data, location, line_data, line_02_data, model_path=None, distance_threshold=1.0):
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
    face_name = knn_clf.predict([face_data])[0]
    if face_name != location_name and closest_distances[0][0][0] > 0.31:
        face_name = "unknown"

    ### version 1.0 ###
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

    X = []
    y = []

    encoding_name = 'faces_encoding.pkl'
    encoding_path = os.path.join(FLAGS.pickle_path, encoding_name)
    model_save_name = 'trained_knn_model.clf'
    model_save_path = os.path.join(FLAGS.pickle_path, model_save_name)
    locations_name = 'all_locations.pkl'
    locations_path = os.path.join(FLAGS.pickle_path, locations_name)
    line_path = os.path.join(FLAGS.pickle_path, 'line.pkl')
    line_02_path = os.path.join(FLAGS.pickle_path, 'line_02.pkl')

    with open(locations_path, 'rb') as f:
        all_locations = pickle.load(f)

    with open(encoding_path, 'rb') as f:
        faces_encoding = pickle.load(f)
    with open(line_02_path, 'rb') as f:
        line_02_data = pickle.load(f)
    with open(line_path, 'rb') as f:
        line_data = pickle.load(f)

    for face_name, face_encoding in faces_encoding:
        X.append(face_encoding)
        y.append(face_name)

    print("Training KNN classifier...")
    classifier = train(X, y, model_save_path=model_save_path, n_neighbors=2)
    print("Training complete!")

    name_list = [i * 100 for i in range(1, 133)]
    count = 0
    stu_names = []
    for name, locations in zip(name_list, all_locations):
        stu_name = []
        image_name = str(name) + '.jpg'
        image_path = os.path.join(FLAGS.images_path, image_name)
        image_data = cv2.imread(image_path)
        image_resized = image_resize(image_data, IMAGE_SIZE)
        image_rgb = convert_bgr_to_rgb(image_resized)
        faces_data = face_recognition.face_encodings(image_rgb, known_face_locations=locations)
        for face_data, location in zip(faces_data, locations):
            face_name = predict(face_data, location, line_data, line_02_data, model_path=model_save_path)
            stu_name.append(face_name)
        print(stu_name)
        stu_names.append(stu_name)
        image_named = draw_op_named(image_rgb, locations, stu_name)
        image_bgr = convert_rgb_to_bgr(image_named)
        save_path = os.path.join(FLAGS.images_save_path, image_name)
        cv2.imwrite(save_path, image_bgr)
        print('processing {0} pic.'.format(count))
        count += 1

    with open('../pickle_path/stu_names.pkl', 'wb') as f:
        pickle.dump(stu_names, f)
    stu_dict = defaultdict(int)

    for num, stu_data in enumerate(stu_names):
        for stu_name_ in stu_data:
            if stu_name_ != 'unknown':
                stu_dict[stu_name_] += 1
        if len(list(stu_dict.keys())) == 37:
            print('37', num)
        if len(list(stu_dict.keys())) == 38:
            print('38', num)
        if len(list(stu_dict.keys())) == 39:
            print('39', num)
        if len(list(stu_dict.keys())) == 40:
            print('40', num)

    for stu_dict_key in stu_dict.keys():
        stu_dict_value = stu_dict[stu_dict_key]
        print('{0}: {1}.'.format(stu_dict_key, stu_dict_value))
