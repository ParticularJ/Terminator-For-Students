import cv2
import argparse
import os


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Process pic')
    parser.add_argument('--input', help='video to process', dest='input', default=None, type=str)
    parser.add_argument('--output', help='pic to store', dest='output', default=None, type=str)

    parser.add_argument('--skip_frame', dest='skip_frame', help='skip number of video', default=100, type=int)
    args = parser.parse_args()
    return args


def process_video(i_video, o_video, num):
    cap = cv2.VideoCapture(i_video)
    num_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    expand_name = '.jpg'
    if not cap.isOpened():
        print("Please check the path.")
    cnt = 0
    while 1:
        ret, frame = cap.read()
        cnt += 1
        if cnt % num == 0 and cnt <= num_frame:
            cv2.imwrite(os.path.join(o_video, str(cnt)+expand_name), frame)
        else:
            break


if __name__ == '__main__':
    args = parse_args()
    if not os.path.exists(args.input):
        os.makedirs(args.input)
    print('Called with args:')
    print(args)
    process_video(args.input, args.output, args.skip_frame)
