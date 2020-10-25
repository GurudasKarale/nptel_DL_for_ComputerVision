import matplotlib.pyplot as plt
import numpy as np
import cv2

image = cv2.imread('.../line.png')

image_copy = np.copy(image)

image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)

gray_img = cv2.cvtColor(image_copy, cv2.COLOR_RGB2GRAY)
gray_img = np.float32(gray_img)
dilst = cv2.cornerHarris(gray_img, 2, 3, 0.04)

dilst = cv2.dilate(dilst,None)

plt.imshow(dilst, cmap='gray')

thresh = 0.1 * dilst.max()

corner_image = np.copy(image_copy)

count = 0
for j in range(0, dilst.shape[0]):
    for i in range(0, dilst.shape[1]):
        if(dilst[j,i] > thresh):
            count += 1
            cv2.circle( corner_image, (i, j), 1, (0,255,0), 1)

print ("Number of Detected Corners: ", count)
plt.imshow(corner_image)
plt.show()
