import argparse
import cv2
import os
import pickle

FLAGS = None


def draw_op(image, face_locations):
    names = range(len(face_locations))
    for (top, right, bottom, left), name in zip(face_locations, names):
        # Draw a box around the face
        cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(image, (left, bottom - 25), (right, bottom), (0, 0, 255), cv2.FILLED)
        # rgb_frame = image[:, :, ::-1]
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(image, str(name), (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)
        cv2.line(image, (723, 2308), (1260, 965), (0, 0, 255), 2)
        cv2.line(image, (723, 2308), (1260, 965), (0, 0, 255), 2)
        cv2.line(image, (723, 2308), (1260, 965), (0, 0, 255), 2)
        cv2.line(image, (723, 2308), (1260, 965), (0, 0, 255), 2)
        cv2.line(image, (723, 2308), (1260, 965), (0, 0, 255), 2)
        cv2.line(image, (723, 2308), (1260, 965), (0, 0, 255), 2)
        cv2.line(image, (723, 2308), (1260, 965), (0, 0, 255), 2)
        cv2.line(image, (723, 2308), (1260, 965), (0, 0, 255), 2)
        print('{0}: ({1}, {2}).'.format(name, (left+right)/2, (top+bottom)/2))
    return image


def draw_op_dict(image, face_locations_dict):
    for key in face_locations_dict.keys():
        for key_ in face_locations_dict[key].keys():
            top, right, bottom, left = face_locations_dict[key][key_]
            cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)
            # Draw a label with a name below the face
            cv2.rectangle(image, (left, bottom - 25), (right, bottom), (0, 0, 255), cv2.FILLED)
            # rgb_frame = image[:, :, ::-1]
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(image, key + '_' + key_, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)
    return image
    #     for num, (top, right, bottom, left) in enumerate(face_locations_dict[key]):
    #         cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)
    #         # Draw a label with a name below the face
    #         cv2.rectangle(image, (left, bottom - 25), (right, bottom), (0, 0, 255), cv2.FILLED)
    #         # rgb_frame = image[:, :, ::-1]
    #         font = cv2.FONT_HERSHEY_DUPLEX
    #         cv2.putText(image, key + '_' + str(num), (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)
    # return image


def image_resize(image, size):
    img_resized = cv2.resize(image, size, interpolation=cv2.INTER_CUBIC)
    print('resize op done.')
    return img_resized


# OpenCV follows BGR order, while matplotlib likely follows RGB order.
def convert_bgr_to_rgb(bgr_img):
    b, g, r = cv2.split(bgr_img)  # get b, g, r
    rgb_img = cv2.merge([r, g, b])  # switch it to rgb
    return rgb_img


def convert_rgb_to_bgr(rgb_img):
    r, g, b = cv2.split(rgb_img)
    bgr_img = cv2.merge([b, g, r])
    return bgr_img


def convert_video_to_images(video_path, images_path, frame_number_per=100):
    input_movie = cv2.VideoCapture(video_path)
    frame_number = 0
    while True:
        # Grab a single frame of video
        ret, frame = input_movie.read()
        frame_number += 1

        # Quit when the input video file ends
        if not ret:
            break

        if frame_number % frame_number_per == 0:
            # # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            # rgb_frame = frame[:, :, ::-1]
            file_name = str(frame_number) + '.jpg'
            file_path = os.path.join(images_path, file_name)
            cv2.imwrite(file_path, frame)
    print('convert_video_to_images op done.')


def parser():
    args = argparse.ArgumentParser()

    args.add_argument('--video_path', default='/home/enningxie/Documents/DataSets/class_video/face/ch02_20180308161036.mp4')
    args.add_argument('--images_path', default='/home/enningxie/Documents/DataSets/face_images/4200.jpg')
    args.add_argument('--to_path', default='/home/enningxie/Documents/DataSets/face_rec/resize_images/4200.jpg')
    return args.parse_args()


def save_op(save_path, save_name, save_data):
    file_path = os.path.join(save_path, save_name)
    with open(file_path, 'wb') as f:
        pickle.dump(save_data, f)
    print('save op done.')


if __name__ == '__main__':
    FLAGS = parser()
    video_path = FLAGS.video_path
    images_path = FLAGS.images_path
    to_path = FLAGS.to_path
    # convert_video_to_images(video_path, images_path)
    # image_resize(images_path, (6144, 3240), to_path)
    input_movie = cv2.VideoCapture(video_path)
    print(input_movie.get(cv2.CAP_PROP_FPS))
