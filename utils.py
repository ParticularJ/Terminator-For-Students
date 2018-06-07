import argparse
import cv2
import os

FLAGS = None


def image_resize(image, size):
    img = cv2.imread(image)
    img_resized = cv2.resize(img, size, interpolation=cv2.INTER_CUBIC)
    print('resize op done.')
    return img_resized



# OpenCV follows BGR order, while matplotlib likely follows RGB order.
def convert_bgr_to_rgb(bgr_img):
    b, g, r = cv2.split(bgr_img)  # get b, g, r
    rgb_img = cv2.merge([r, g, b])  # switch it to rgb
    return rgb_img


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


if __name__ == '__main__':
    FLAGS = parser()
    video_path = FLAGS.video_path
    images_path = FLAGS.images_path
    to_path = FLAGS.to_path
    # convert_video_to_images(video_path, images_path)
    # image_resize(images_path, (6144, 3240), to_path)
    input_movie = cv2.VideoCapture(video_path)
    print(input_movie.get(cv2.CAP_PROP_FPS))