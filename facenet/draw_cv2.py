import cv2
import face_recognition as fr
import pickle
from utils import *
pkl_path = '/home/enningxie/Documents/DataSets/face_rec/pkl_images/10600.pkl'
image_path = '/home/enningxie/Documents/DataSets/face_images/10600.jpg'


image_data = cv2.imread(image_path)
image_resized = image_resize(image_data, (6553, 3456))
image_rgb = convert_bgr_to_rgb(image_resized)

with open(pkl_path, 'rb') as f:
    locations = pickle.load(f)

# cv2.rectangle(image_rgb, (0, ), (100, 100), (255, 0, 0), 2)


cv2.rectangle(image_rgb, (4631, 807), (4785, 962), (255, 0, 0), 2)



image_bgr = convert_rgb_to_bgr(image_rgb)

# new_img = cv2.resize(image_bgr, (1920, 1080))
# cv2.namedWindow("Image")
# cv2.imshow("Image", new_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
cv2.imwrite('./test_cv2_02.jpg', image_bgr[807:962, 4631:4785])


print(len(locations))

print(sorted(locations[:3], key=lambda x: x[1]+x[2]))




