import cv2
import numpy as np

# 4.1 Resize
img = cv2.imread('./Q4_Image/SQUARE-01.png')
res_img = cv2.resize(img, (256,256), interpolation=cv2.INTER_AREA)
cv2.imshow('Resize', res_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 4.2 Translation
tran_mat = np.float32([[1,0,0],[0,1,60]])
translation = cv2.warpAffine(res_img, tran_mat, (400, 300))
cv2.imshow('Translation', translation)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 4.3 Rotation, Scaling
rot_mat = cv2.getRotationMatrix2D((128,188), 10, 0.5)
rotation = cv2.warpAffine(res_img, rot_mat, (400,300))
cv2.imshow('Rotation, Scaling', rotation)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 4.4 Shearing
old = np.float32([[50,50],[200,50],[50,200]])
new = np.float32([[10,100],[200,50],[100,250]])
shearing_mat = cv2.getAffineTransform(old, new)
shearing = cv2.warpAffine(rotation, shearing_mat, (400,300))
cv2.imshow('Shearing', shearing)
cv2.waitKey(0)
cv2.destroyAllWindows()
