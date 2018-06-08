import argparse
import face_recognition as fr
import cv2
import os
from utils import *


FLAGS = None


def parser():
    args = argparse.ArgumentParser()
    args.add_argument('--image_path', default='/var/Data/xz/face/face_images')
    args.add_argument('--save_path', default='/var/Data/xz/face/process_images')
    args.add_argument('--saved_path', default='/var/Data/xz/face/saved_images')
    return args.parse_args()


def find_faces_in_pic(image_path, save_path, name_list):
    face_locations_list = []
    for name in name_list:
        image_name = str(name)+'.jpg'
        image_path_ = os.path.join(image_path, image_name)
        image_data = cv2.imread(image_path_)
        image_resized = image_resize(image_data, (6553, 3456))
        image_rgb = convert_bgr_to_rgb(image_resized)
        # image = fr.load_image_file(image_path)
        face_locations = fr.face_locations(image_rgb, number_of_times_to_upsample=0, model="cnn")
        face_locations_list.append(face_locations)
        image = draw_op(image_rgb, face_locations)
        image_bgr = convert_rgb_to_bgr(image)
        save_path_ = os.path.join(save_path, image_name)
        cv2.imwrite(save_path_, image_bgr)
    print('detect op end.')
    return face_locations_list


def count_in_pic(img_path):
    images_list = os.listdir(img_path)
    face_locations_list = []
    face_locations_count = []
    for image in images_list:
        image_path = os.path.join(img_path, image)
        image_data = cv2.imread(image_path)
        image_resized = image_resize(image_data, (6553, 3456))
        image_rgb = convert_bgr_to_rgb(image_resized)
        face_locations = fr.face_locations(image_rgb, number_of_times_to_upsample=0, model="cnn")
        face_locations_list.append(face_locations)
        face_locations_count.append(len(face_locations))
    return images_list, face_locations_count, face_locations_list


if __name__ == '__main__':
    FLAGS = parser()
    image_path = FLAGS.image_path
    save_path = FLAGS.save_path
    saved_path = FLAGS.saved_path
    #images_list, face_locations_count, face_locations_list = count_in_pic(image_path)
    #max_num = max(face_locations_count)
    #for i, count in zip(images_list, face_locations_count):
        #if count == max_num:
            #print('{0} has {1} faces.'.format(i, count))
    face_locations_list = find_faces_in_pic(image_path, save_path, [10600])
    save_name = '10600.pkl'
    save_op(saved_path, save_name, face_locations_list[0])




