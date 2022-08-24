import cv2
import numpy as np

# 1.1 load image
img_1 = cv2.imread('./Q1_Image/Sun.jpg')
height, width, _ = img_1.shape
print('height: {}'.format(height))
print('width: {}'.format(width))
cv2.imshow('Hw1-1', img_1)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 1.2 color separation
b, g, r = cv2.split(img_1)

zeros = np.zeros(img_1.shape[:2], dtype="uint8")
B_channel = cv2.merge([b, zeros, zeros])
G_channel = cv2.merge([zeros, g, zeros])
R_channel = cv2.merge([zeros, zeros, r])
cv2.imshow('B channel', B_channel)
cv2.imshow('G channel', G_channel)
cv2.imshow('R channel', R_channel)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 1.3 color transformation
img_2 = cv2.merge([b, g, r])
img_12gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)

row, col = img_2.shape[0:2]
for i in range(row):
    for j in range(col):
        img_2[i, j] = sum(img_2[i, j])/3

cv2.imshow('l1', img_12gray)
cv2.imshow('l2', img_2)
cv2.waitKey(0)
cv2.destroyAllWindows()


