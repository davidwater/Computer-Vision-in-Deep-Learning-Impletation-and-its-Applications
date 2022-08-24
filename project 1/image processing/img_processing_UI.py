# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Hw1_F44076241_水修澤_V1.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!
import cv2
import numpy as np
import skimage.exposure
from skimage.exposure import rescale_intensity
from PyQt5 import QtCore, QtGui, QtWidgets


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

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1106, 520)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(40, 80, 221, 371))
        self.groupBox.setObjectName("groupBox")
        self.pushButton = QtWidgets.QPushButton(self.groupBox)
        self.pushButton.setGeometry(QtCore.QRect(20, 40, 181, 61))
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_2.setGeometry(QtCore.QRect(20, 120, 181, 61))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_3 = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_3.setGeometry(QtCore.QRect(20, 200, 181, 61))
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_4 = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_4.setGeometry(QtCore.QRect(20, 280, 181, 61))
        self.pushButton_4.setObjectName("pushButton_4")
        self.groupBox_2 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_2.setGeometry(QtCore.QRect(290, 80, 221, 371))
        self.groupBox_2.setObjectName("groupBox_2")
        self.pushButton_5 = QtWidgets.QPushButton(self.groupBox_2)
        self.pushButton_5.setGeometry(QtCore.QRect(20, 70, 181, 61))
        self.pushButton_5.setObjectName("pushButton_5")
        self.pushButton_6 = QtWidgets.QPushButton(self.groupBox_2)
        self.pushButton_6.setGeometry(QtCore.QRect(20, 150, 181, 61))
        self.pushButton_6.setObjectName("pushButton_6")
        self.pushButton_7 = QtWidgets.QPushButton(self.groupBox_2)
        self.pushButton_7.setGeometry(QtCore.QRect(20, 240, 181, 61))
        self.pushButton_7.setObjectName("pushButton_7")
        self.groupBox_3 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_3.setGeometry(QtCore.QRect(550, 80, 241, 371))
        self.groupBox_3.setObjectName("groupBox_3")
        self.pushButton_8 = QtWidgets.QPushButton(self.groupBox_3)
        self.pushButton_8.setGeometry(QtCore.QRect(30, 40, 181, 61))
        self.pushButton_8.setObjectName("pushButton_8")
        self.pushButton_9 = QtWidgets.QPushButton(self.groupBox_3)
        self.pushButton_9.setGeometry(QtCore.QRect(30, 120, 181, 61))
        self.pushButton_9.setObjectName("pushButton_9")
        self.pushButton_10 = QtWidgets.QPushButton(self.groupBox_3)
        self.pushButton_10.setGeometry(QtCore.QRect(30, 200, 181, 61))
        self.pushButton_10.setObjectName("pushButton_10")
        self.pushButton_11 = QtWidgets.QPushButton(self.groupBox_3)
        self.pushButton_11.setGeometry(QtCore.QRect(30, 280, 181, 61))
        self.pushButton_11.setObjectName("pushButton_11")
        self.groupBox_4 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_4.setGeometry(QtCore.QRect(820, 80, 231, 371))
        self.groupBox_4.setObjectName("groupBox_4")
        self.pushButton_12 = QtWidgets.QPushButton(self.groupBox_4)
        self.pushButton_12.setGeometry(QtCore.QRect(20, 40, 181, 61))
        self.pushButton_12.setObjectName("pushButton_12")
        self.pushButton_13 = QtWidgets.QPushButton(self.groupBox_4)
        self.pushButton_13.setGeometry(QtCore.QRect(20, 120, 181, 61))
        self.pushButton_13.setObjectName("pushButton_13")
        self.pushButton_14 = QtWidgets.QPushButton(self.groupBox_4)
        self.pushButton_14.setGeometry(QtCore.QRect(20, 200, 181, 61))
        self.pushButton_14.setObjectName("pushButton_14")
        self.pushButton_15 = QtWidgets.QPushButton(self.groupBox_4)
        self.pushButton_15.setGeometry(QtCore.QRect(20, 280, 181, 61))
        self.pushButton_15.setObjectName("pushButton_15")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1106, 25))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.pushButton.clicked.connect(self.load_image)
        self.pushButton_2.clicked.connect(self.color_separation)
        self.pushButton_3.clicked.connect(self.color_transformation)
        self.pushButton_4.clicked.connect(self.blending)
        self.pushButton_5.clicked.connect(self.Gaussian_Blur)
        self.pushButton_6.clicked.connect(self.Bilateral_Filter)
        self.pushButton_7.clicked.connect(self.Median_Filter)
        self.pushButton_8.clicked.connect(self.gaussian_blur)
        self.pushButton_9.clicked.connect(self.sobel_x)
        self.pushButton_10.clicked.connect(self.sobel_y)
        self.pushButton_11.clicked.connect(self.magnitude)
        self.pushButton_12.clicked.connect(self.resize)
        self.pushButton_13.clicked.connect(self.translation)
        self.pushButton_14.clicked.connect(self.rotation_scaling)
        self.pushButton_15.clicked.connect(self.shearing)
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def load_image(self):
        # 1.1 load image
        img_1 = cv2.imread('.\Q1_Image\Sun.jpg')
        height, width, _ = img_1.shape
        print('height: {}'.format(height))
        print('width: {}'.format(width))
        cv2.imshow('Hw1-1', img_1)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def color_separation(self):
        # 1.2 color separation
        img_1 = cv2.imread('./Q1_Image/Sun.jpg')
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

    def color_transformation(self):
        # 1.3 color transformation
        img_1 = cv2.imread('./Q1_Image/Sun.jpg')
        b, g, r = cv2.split(img_1)
        img_2 = cv2.merge([b, g, r])
        img_12gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)

        row, col = img_2.shape[0:2]
        for i in range(row):
            for j in range(col):
                img_2[i, j] = sum(img_2[i, j]) / 3

        cv2.imshow('l1', img_12gray)
        cv2.imshow('l2', img_2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def blending(self):
        # 1.4 blending
        img_3 = cv2.imread('./Q1_Image/Dog_Strong.jpg')
        img_4 = cv2.imread('./Q1_Image/Dog_Weak.jpg')
        blend_1 = cv2.addWeighted(img_3, 0.5, img_4, 0.5, 0)
        cv2.imshow('Blend', blend_1)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # update input value to blend pictures
        def update_Blend(x):
            alpha = x / 100
            beta = 1 - alpha
            blend_2 = cv2.addWeighted(img_3, alpha, img_4, beta, 0)
            cv2.imshow('Blend', blend_2)

        cv2.namedWindow('Blend')
        cv2.createTrackbar('weights', 'Blend', 0, 100, update_Blend)

        update_Blend(0)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def Gaussian_Blur(self):
        img1 = cv2.imread('./Q2_Image/Lenna_whiteNoise.jpg')
        Gaussian_blur = cv2.GaussianBlur(img1, (5, 5), 0)
        cv2.imshow('Gaussian Filter', Gaussian_blur)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def Bilateral_Filter(self):
        img1 = cv2.imread('./Q2_Image/Lenna_whiteNoise.jpg')
        Bilateral_blur = cv2.bilateralFilter(img1, 9, 90, 90)
        cv2.imshow('Bilateral Filter', Bilateral_blur)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def Median_Filter(self):
        img2 = cv2.imread('./Q2_Image/Lenna_pepperSalt.jpg')
        Median_blur_3 = cv2.medianBlur(img2, 3)
        Median_blur_5 = cv2.medianBlur(img2, 5)
        cv2.imshow('Median Filter 3x3', Median_blur_3)
        cv2.imshow('Median Filter 5x5', Median_blur_5)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    def gaussian_blur(self):
        # 3.1 Gaussian Blur (without opencv)
        img = cv2.imread('./Q3_Image/House.jpg')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imshow('Grayscale', gray)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # create 3*3 Gaussian Filter
        x, y = np.mgrid[-1:2, -1:2]
        gaussian_filter = np.exp(-(x ** 2 + y ** 2))
        gaussian_filter = gaussian_filter / gaussian_filter.sum()

        gaussian_blur = convolve(gray, gaussian_filter)
        cv2.imshow('Gaussian Blur', gaussian_blur)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    def sobel_x(self):
        # 3.2 Sobel X
        # construct the Sobel x-axis kernel
        img = cv2.imread('./Q3_Image/House.jpg')
        sobelX = np.array((
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]), dtype="int")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        x, y = np.mgrid[-1:2, -1:2]
        gaussian_filter = np.exp(-(x ** 2 + y ** 2))
        gaussian_filter = gaussian_filter / gaussian_filter.sum()

        gaussian_blur = convolve(gray, gaussian_filter)
        sobel_x = convolve(gaussian_blur, sobelX)
        cv2.imshow('Sobel X', sobel_x)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    def sobel_y(self):
        # 3.3 Sobel Y
        # construct the Sobel y-axis kernel
        img = cv2.imread('./Q3_Image/House.jpg')
        sobelY = np.array((
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]), dtype="int")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        x, y = np.mgrid[-1:2, -1:2]
        gaussian_filter = np.exp(-(x ** 2 + y ** 2))
        gaussian_filter = gaussian_filter / gaussian_filter.sum()

        gaussian_blur = convolve(gray, gaussian_filter)
        sobel_y = convolve(gaussian_blur, sobelY)
        cv2.imshow('Sobel Y', sobel_y)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def magnitude(self):
        sobelX = np.array((
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]), dtype="int")
        img = cv2.imread('./Q3_Image/House.jpg')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        x, y = np.mgrid[-1:2, -1:2]
        gaussian_filter = np.exp(-(x ** 2 + y ** 2))
        gaussian_filter = gaussian_filter / gaussian_filter.sum()

        gaussian_blur = convolve(gray, gaussian_filter)
        sobel_x = convolve(gaussian_blur, sobelX)
        sobelY = np.array((
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]), dtype="int")
        img = cv2.imread('./Q3_Image/House.jpg')
        sobel_y = convolve(gaussian_blur, sobelY)
        magnitude = np.hypot(sobel_x, sobel_y)
        magnitude = skimage.exposure.rescale_intensity(magnitude, in_range='image', out_range=(0, 255)).astype("uint8")
        cv2.imshow('Magnitude', magnitude)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    def resize(self):
        img = cv2.imread('./Q4_Image/SQUARE-01.png')
        res_img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
        cv2.imshow('Resize', res_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def translation(self):
        img = cv2.imread('./Q4_Image/SQUARE-01.png')
        res_img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
        tran_mat = np.float32([[1, 0, 0], [0, 1, 60]])
        translation = cv2.warpAffine(res_img, tran_mat, (400, 300))
        cv2.imshow('Translation', translation)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def rotation_scaling(self):
        img = cv2.imread('./Q4_Image/SQUARE-01.png')
        res_img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
        rot_mat = cv2.getRotationMatrix2D((128, 188), 10, 0.5)
        rotation = cv2.warpAffine(res_img, rot_mat, (400, 300))
        cv2.imshow('Rotation, Scaling', rotation)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def shearing(self):
        img = cv2.imread('./Q4_Image/SQUARE-01.png')
        res_img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
        rot_mat = cv2.getRotationMatrix2D((128, 188), 10, 0.5)
        rotation = cv2.warpAffine(res_img, rot_mat, (400, 300))
        old = np.float32([[50, 50], [200, 50], [50, 200]])
        new = np.float32([[10, 100], [200, 50], [100, 250]])
        shearing_mat = cv2.getAffineTransform(old, new)
        shearing = cv2.warpAffine(rotation, shearing_mat, (400, 300))
        cv2.imshow('Shearing', shearing)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.groupBox.setTitle(_translate("MainWindow", "1. Image Processing"))
        self.pushButton.setText(_translate("MainWindow", "1.1 Load Image"))
        self.pushButton_2.setText(_translate("MainWindow", "1.2  Color Separation"))
        self.pushButton_3.setText(_translate("MainWindow", "1.3 Color Transformation"))
        self.pushButton_4.setText(_translate("MainWindow", "1.4 Blending"))
        self.groupBox_2.setTitle(_translate("MainWindow", "2. Image Smoothing "))
        self.pushButton_5.setText(_translate("MainWindow", "2.1 Gaussian Blur"))
        self.pushButton_6.setText(_translate("MainWindow", "2.2 Bilateral Filter"))
        self.pushButton_7.setText(_translate("MainWindow", "2.3 Median Filter"))
        self.groupBox_3.setTitle(_translate("MainWindow", "3. Edge Detection"))
        self.pushButton_8.setText(_translate("MainWindow", "3.1 Gaussian Blur"))
        self.pushButton_9.setText(_translate("MainWindow", "3.2 Sobel X"))
        self.pushButton_10.setText(_translate("MainWindow", "3.3 Sobel Y"))
        self.pushButton_11.setText(_translate("MainWindow", "3.4 Magnitude"))
        self.groupBox_4.setTitle(_translate("MainWindow", "4. Transformation"))
        self.pushButton_12.setText(_translate("MainWindow", "4.1 Resize"))
        self.pushButton_13.setText(_translate("MainWindow", "4.2 Translation"))
        self.pushButton_14.setText(_translate("MainWindow", "4.3 Rotatoin, Scaling"))
        self.pushButton_15.setText(_translate("MainWindow", "4.4 Shearing"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

