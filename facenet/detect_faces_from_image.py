import argparse
import face_recognition as fr
import cv2


FLAGS = None


def parser():
    args = argparse.ArgumentParser()
    args.add_argument('--image_path', default='../data/4200.jpg')
    args.add_argument('--save_path', default='../processed_data/4200_02.jpg')
    return args.parse_args()


def find_faces_in_pic(image_path, save_path):
    image = fr.load_image_file(image_path)
    face_locations = fr.face_locations(image, number_of_times_to_upsample=0, model="cnn")
    for top, right, bottom, left in face_locations:
        # Draw a box around the face
        cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(image, (left, bottom - 25), (right, bottom), (0, 0, 255), cv2.FILLED)
        rgb_frame = image[:, :, ::-1]
    cv2.imwrite(save_path, rgb_frame)
    print('detect op end.')


if __name__ == '__main__':
    FLAGS = parser()
    image_path = FLAGS.image_path
    save_path = FLAGS.save_path
    find_faces_in_pic(image_path, save_path)


