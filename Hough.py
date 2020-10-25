import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('.../line.png', cv2.IMREAD_COLOR)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

edges = cv2.Canny(gray, 50, 200)

lines = cv2.HoughLinesP(edges, 1, np.pi/180, 15, minLineLength=60,maxLineGap=250)

for line in lines:
    x1,y1,x2,y2 = line[0]
    print ("start and end coordinates: ", (x1,y1),(x2,y2) )
    cv2.line(img, (x1,y1), (x2,y2), (255,0,0), 3)

plt.imshow(img)
plt.title('Detected Line Image')
plt.show()
