import argparse
import face_recognition as fr
import cv2
import os
from utils import image_resize, convert_bgr_to_rgb, draw_op


FLAGS = None


def parser():
    args = argparse.ArgumentParser()
    args.add_argument('--image_path', default='/var/Data/xz/face/face_images')
    args.add_argument('--save_path', default='/var/Data/xz/face/process_images')
    return args.parse_args()


def find_faces_in_pic(image_path, save_path, name_list):
    for name in name_list:
        image_name = str(name)+'.jpg'
        image_path_ = os.path.join(image_path, image_name)
        image_data = cv2.imread(image_path_)
        image_resized = image_resize(image_data, (6144, 3240))
        image_rgb = convert_bgr_to_rgb(image_resized)
        # image = fr.load_image_file(image_path)
        face_locations = fr.face_locations(image_rgb, number_of_times_to_upsample=0, model="cnn")
        image = draw_op(image_data, face_locations)
        save_path_ = os.path.join(save_path, image_name)
        cv2.imwrite(save_path_, image)
    print('detect op end.')


def count_in_pic(img_path):
    images_list = os.listdir(img_path)
    face_locations_list = []
    face_locations_count = []
    for image in images_list:
        image_path = os.path.join(img_path, image)
        image_resized = image_resize(image_path, (6144, 3240))
        image_rgb = convert_bgr_to_rgb(image_resized)
        face_locations = fr.face_locations(image_rgb, number_of_times_to_upsample=0, model="cnn")
        face_locations_list.append(face_locations)
        face_locations_count.append(len(face_locations))
    return images_list, face_locations_count, face_locations_list


if __name__ == '__main__':
    FLAGS = parser()
    image_path = FLAGS.image_path
    save_path = FLAGS.save_path
    # images_list, face_locations_count, face_locations_list = count_in_pic(image_path)
    # max_num = max(face_locations_count)
    # for i, count in zip(images_list, face_locations_count):
    #     if count == max_num:
    #         print('{0} has {1} faces.'.format(i, count))
    find_faces_in_pic(image_path, save_path, [12900, 6600, 5800, 12400, 12300, 5900, 10600])


