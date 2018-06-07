import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import convert_bgr_to_rgb
# OpenCV follows BGR order, while matplotlib likely follows RGB order.


img_path = '/home/enningxie/Documents/DataSets/face_rec/san_francisco.jpg'

bgr_img = cv2.imread(img_path)
rgb_img = convert_bgr_to_rgb(bgr_img)
# gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
# cv2.imwrite('/home/enningxie/Documents/DataSets/face_rec/san_francisco_gray.jpg', gray_img)

# plt.imshow(bgr_img)
plt.imshow(rgb_img)
plt.xticks([])
plt.yticks([])  # to hide tick values on X and Y axis
plt.show()

while True:
    k = cv2.waitKey(0) & 0xFF
    if k == 27:
        break  # code for the ESC key

cv2.destroyAllWindows()
