import argparse
from utils import convert_video_to_images, image_resize, convert_bgr_to_rgb, draw_op_named, convert_rgb_to_bgr
import os
import cv2
import face_recognition as fr
from faces_named import predict

FLAGS = None
IMAGE_SIZE = (6553, 3456)


def parser():
    args = argparse.ArgumentParser()
    args.add_argument('--video_path', default='/var/Data/xz/face/video_path/ch02_20180308163706.mp4')
    args.add_argument('--images_path', default='/var/Data/xz/face/images_path/ch02_20180308163706')
    args.add_argument('--rec_path', default='/var/Data/xz/face/rec_path/ch02_20180308163706')
    args.add_argument('--model_path', default='/var/Data/xz/face/pickle_path/trained_knn_model.clf')
    return args.parse_args()


def count_in_pic(images_rgb):
    face_locations_list = []
    for image_rgb in images_rgb:
        face_locations = fr.face_locations(image_rgb, number_of_times_to_upsample=0, model="cnn")
        face_locations_list.append(face_locations)
    return face_locations_list


def _images_rgb(images_path):
    images_list = os.listdir(images_path)
    images_rgb = []
    for image in images_list:
        image_path = os.path.join(images_path, image)
        image_data = cv2.imread(image_path)
        image_resized = image_resize(image_data, IMAGE_SIZE)
        image_rgb = convert_bgr_to_rgb(image_resized)
        images_rgb.append(image_rgb)
    return images_list, images_rgb


def _save_result_op(images_rgb, face_locations_list, images_list, model_path, rec_path):
    count = 0
    for image_rgb, locations, image_name in zip(images_rgb, face_locations_list, images_list):
        stu_name = []
        faces_data = fr.face_encodings(image_rgb, known_face_locations=locations)
        for face_data in faces_data:
            face_name = predict(face_data, model_path=model_path)
            stu_name.append(face_name)
        print(stu_name)
        image_named = draw_op_named(image_rgb, locations, stu_name)
        image_bgr = convert_rgb_to_bgr(image_named)
        save_path = os.path.join(rec_path, image_name)
        cv2.imwrite(save_path, image_bgr)
        print('processing {0} pic.'.format(count))
        count += 1


if __name__ == '__main__':
    FLAGS = parser()
    video_path = FLAGS.video_path
    images_path = FLAGS.images_path
    rec_path = FLAGS.rec_path
    model_path = FLAGS.model_path
    convert_video_to_images(video_path, images_path)
    images_list, images_rgb = _images_rgb(images_path)
    face_locations_list = count_in_pic(images_rgb)
    _save_result_op(images_rgb, face_locations_list, images_list, model_path, rec_path)