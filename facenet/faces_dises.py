import argparse
import os
from collections import defaultdict
import face_recognition as fr
FLAGS = None


def parser():
    args = argparse.ArgumentParser()
    args.add_argument('--pick_path', default='/home/enningxie/Documents/DataSets/face_rec/pick_face')
    return args.parse_args()


if __name__ == '__main__':
    FLAGS = parser()
    stu_names = []
    faces_encoding = defaultdict()
    faces_dis = defaultdict()
    for sub_name in os.listdir(FLAGS.pick_path):
        sub_path = os.path.join(FLAGS.pick_path, sub_name)
        for image_name in os.listdir(sub_path):
            image_path = os.path.join(sub_path, image_name)
            image_data = fr.load_image_file(image_path)
            face_location = fr.face_locations(image_data, number_of_times_to_upsample=0, model='cnn')
            face_encoding = fr.face_encodings(image_data, known_face_locations=face_location)
            faces_encoding[sub_name] = face_encoding[0]
    # print(len(list(faces_encoding.keys())))
    sorted_names = sorted(list(faces_encoding.keys()))
    for i in range(len(sorted_names)):
        for j in range(i+1, len(sorted_names)):
            dis_key = sorted_names[i] + 'vs' + sorted_names[j]
            faces_dis[dis_key] = fr.face_distance([faces_encoding[sorted_names[i]]], faces_encoding[sorted_names[j]])[0]
    # for key in faces_dis.keys():
    #     faces_dis_value = faces_dis[key]
    #     print('faces_dis {0}: {1}.'.format(key, faces_dis_value))

    faces_dis_values = list(faces_dis.values())
    faces_dis_keys = list(faces_dis.keys())

    faces_dis_new = []

    for key_, value_ in zip(faces_dis_keys, faces_dis_values):
        faces_dis_new.append((key_, value_))

    faces_dis_sorted = sorted(faces_dis_new, key=lambda x: x[1])

    for key__, value__ in faces_dis_sorted:
        print('faces_dis {0}: {1}.'.format(key__, value__))