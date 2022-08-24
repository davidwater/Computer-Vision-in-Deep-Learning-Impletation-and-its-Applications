import cv2
import numpy as np
import skimage.exposure
from skimage.exposure import rescale_intensity


def convolve(image, kernel):
	# grab the spatial dimensions of the image, along with the spatial dimensions of the kernel
	(iH, iW) = image.shape[:2]
	(kH, kW) = kernel.shape[:2]
	# allocate memory for the output image, taking care to
	# "pad" the borders of the input image so the spatial size (i.e., width and height) are not reduced
	pad = (kW - 1) // 2
	image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
	output = np.zeros((iH, iW), dtype="float32")
	# loop over the input image, "sliding" the kernel across
	# each (x, y)-coordinate from left-to-right and top to
	# bottom
	for y in np.arange(pad, iH + pad):
		for x in np.arange(pad, iW + pad):
			# extract the ROI of the image by extracting the
			# *center* region of the current (x, y)-coordinates
			# dimensions
			roi = image[y - pad:y + pad + 1, x - pad:x + pad + 1]
			# perform the actual convolution by taking the
			# element-wise multiplicate between the ROI and
			# the kernel, then summing the matrix
			k = (roi * kernel).sum()
			# store the convolved value in the output (x,y)-
			# coordinate of the output image
			output[y - pad, x - pad] = k

	# rescale the output image to be in the range [0, 255]
	output = rescale_intensity(output, in_range=(0, 255))
	output = (output * 255).astype("uint8")

	# return the output image
	return output


# 3.1 Gaussian Blur (without opencv)
img = cv2.imread('./Q3_Image/House.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('Grayscale', gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

# create 3*3 Gaussian Filter
x, y = np.mgrid[-1:2, -1:2]
gaussian_filter = np.exp(-(x**2+y**2))
gaussian_filter = gaussian_filter / gaussian_filter.sum()

gaussian_blur = convolve(gray, gaussian_filter)
cv2.imshow('Gaussian Blur', gaussian_blur)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 3.2 Sobel X
# construct the Sobel x-axis kernel
sobelX = np.array((
	[-1, 0, 1],
	[-2, 0, 2],
	[-1, 0, 1]), dtype="int")
sobel_x = convolve(gaussian_blur, sobelX)
cv2.imshow('Sobel X', sobel_x)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 3.3 Sobel Y
# construct the Sobel y-axis kernel
sobelY = np.array((
	[-1, -2, -1],
	[0, 0, 0],
	[1, 2, 1]), dtype="int")
sobel_y = convolve(gaussian_blur, sobelY)
cv2.imshow('Sobel Y', sobel_y)
cv2.waitKey(0)
cv2.destroyAllWindows()


# 3.4 Magnitude
magnitude = np.hypot(sobel_x, sobel_y)
magnitude = skimage.exposure.rescale_intensity(magnitude, in_range='image', out_range=(0, 255)).astype("uint8")
cv2.imshow('Magnitude', magnitude)
cv2.waitKey(0)
cv2.destroyAllWindows()