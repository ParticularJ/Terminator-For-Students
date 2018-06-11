import argparse
import face_recognition as fr
import os
import numpy as np
from collections import defaultdict
FLAGS = None


def parser():
    args = argparse.ArgumentParser()
    args.add_argument('--faces_folder', default='/home/enningxie/Documents/DataSets/face_rec/faces_path/4_7')
    return args.parse_args()


def best_image_num(faces_encoding, files_name):
    for i in range(len(faces_encoding)):
        face_name_now = files_name[i]
        face_encoding_now = faces_encoding[i][0]
        for j in range(i+1, len(faces_encoding)):
            dis = fr.face_distance([face_encoding_now], faces_encoding[j][0])
            if dis[0] < 0.3:
                face_dis_dict[face_name_now] += 1
                face_dis_dict[files_name[j]] += 1

    face_dis_list = list(face_dis_dict.values())
    face_dis_np = np.asarray(face_dis_list)
    max_index = np.argmax(face_dis_np)
    face_dis_list_ = list(face_dis_dict.keys())
    for key in face_dis_dict.keys():
        value = face_dis_dict[key]
        print('{0}: {1}.'.format(key, value))

    print(face_dis_list[max_index])
    print(face_dis_list_[max_index])


if __name__ == '__main__':
    FLAGS = parser()
    files_name = []
    faces_encoding = []
    distance = []
    face_dis_dict = defaultdict(int)
    for file_name in os.listdir(FLAGS.faces_folder):
        file_path = os.path.join(FLAGS.faces_folder, file_name)
        image = fr.load_image_file(file_path)
        face_location = fr.face_locations(image, number_of_times_to_upsample=0, model='cnn')
        if len(face_location) < 1:
            print('{0} detect none.'.format(file_name))
        else:
            files_name.append(file_name)
            face_encoding = fr.face_encodings(image, known_face_locations=face_location)
            faces_encoding.append(face_encoding)
    # count = 0
    # for name, encoding in zip(files_name, faces_encoding):
    #     dis = fr.face_distance([faces_encoding[47][0]], encoding[0])
    #     print('{0}: dis between {1} and {2} is {3}.'.format(count, '3700.jpg', name, dis))
    #     distance.extend(dis)
    #     count += 1
    # distance_np = np.asarray(distance)
    # print(files_name[np.argmax(distance_np)])
    best_image_num(faces_encoding, files_name)
