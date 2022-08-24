import cv2
import numpy as np

# 2.1 Gaussian Blur
img1 = cv2.imread('./Q2_Image/Lenna_whiteNoise.jpg')
Gaussian_blur = cv2.GaussianBlur(img1, (5,5), 0)
cv2.imshow('Gaussian Filter', Gaussian_blur)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 2.2 Bilateral Filter
Bilateral_blur = cv2.bilateralFilter(img1, 9, 90, 90)
cv2.imshow('Bilateral Filter', Bilateral_blur)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 2.3 Median Filter
img2 = cv2.imread('./Q2_Image/Lenna_pepperSalt.jpg')
Median_blur_3 = cv2.medianBlur(img2, 3)
Median_blur_5 = cv2.medianBlur(img2, 5)
cv2.imshow('Median Filter 3x3', Median_blur_3)
cv2.imshow('Median Filter 5x5', Median_blur_5)
cv2.waitKey(0)
cv2.destroyAllWindows()

