import cv2
import pickle
from utils import *

pkl_path = '/home/enningxie/Documents/DataSets/face_rec/pkl_images/10600.pkl'
image_path = '/home/enningxie/Documents/DataSets/face_images/10600.jpg'

with open(pkl_path, 'rb') as f:
    locations = pickle.load(f)

image_data = cv2.imread(image_path)
image_reshape = image_resize(image_data, (6553, 3456))
sorted_locations = sorted(locations, key=lambda x: (x[2], x[1]))
image = draw_op(image_reshape, sorted_locations)

cv2.imwrite('./test_cv2_02.jpg', image)