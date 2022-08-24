# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Hw2_Q5_F44076241_水修澤_V1.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!
from keras.applications.resnet_v2 import ResNet50V2
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, merge, Input, Concatenate, add
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D, GlobalAveragePooling2D
from keras.utils import np_utils
from keras.models import model_from_json
from keras import backend as K
from keras.preprocessing import image
from tensorflow.keras.optimizers import Adam
from keras.utils.data_utils import get_file
import random
import os
import cv2
import keras

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import datetime, os
from PyQt5 import QtCore, QtGui, QtWidgets


def get_image(index):
    img_width = 224
    img_height = 224
    img_size = (img_width, img_height)
    img = cv2.imread('./test/%d.jpg' % index)
    img = cv2.resize(img, img_size)
    img.astype(np.float32) / 255
    return img

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(734, 534)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(190, 30, 301, 361))
        self.groupBox.setObjectName("groupBox")
        self.pushButton = QtWidgets.QPushButton(self.groupBox)
        self.pushButton.setGeometry(QtCore.QRect(60, 80, 171, 28))
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_2.setGeometry(QtCore.QRect(60, 140, 171, 28))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_3 = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_3.setGeometry(QtCore.QRect(60, 200, 171, 28))
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_4 = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_4.setGeometry(QtCore.QRect(60, 310, 171, 28))
        self.pushButton_4.setObjectName("pushButton_4")
        self.lineEdit = QtWidgets.QLineEdit(self.groupBox)
        self.lineEdit.setGeometry(QtCore.QRect(60, 260, 171, 22))
        self.lineEdit.setObjectName("lineEdit")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 734, 25))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.pushButton.clicked.connect(self.show_structure)
        self.pushButton_2.clicked.connect(self.show_tensorboard)
        self.pushButton_3.clicked.connect(self.test)
        self.pushButton_4.clicked.connect(self.data_augmentation)
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)


    def show_structure(self):
        net = ResNet50V2(include_top=False, weights='imagenet', input_tensor=None, input_shape=(224, 224, 3))
        x = net.output
        x = AveragePooling2D((7, 7), name='avg_pool')(x)
        x = Flatten()(x)

        x = Dropout(0.5)(x)

        output_layer = Dense(2, activation='softmax', name='softmax')(x)

        base_model = Model(inputs=net.input, outputs=output_layer)

        for layer in base_model.layers[:2]:
            layer.trainable = False
        for layer in base_model.layers[2:]:
            layer.trainable = True

        lr = 0.00001
        base_model.compile(loss='binary_crossentropy', optimizer=Adam(lr), metrics=['accuracy'])
        base_model.summary()

        #base_model.fit_generator(
            #train_generator,
            #steps_per_epoch=train_generator.samples // 8,
            #validation_data=validation_generator,
            #validation_steps=validation_generator.samples // 8,
            #epochs=5,
            #callbacks=[tensorboard_callback])


    def show_tensorboard(self):
        epoch_acc_loss = cv2.imread('final_result.png')
        cv2.imshow('Accuracy', epoch_acc_loss)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    def test(self):
        pre_train_model = keras.models.load_model('ResNet50_final.h5')
        data = self.lineEdit.text()

        plt.figure()
        if (int(data) - 1 > -1 and int(data) -1 < 2500):
            x = get_image(int(data))
            test_img = plt.imread('./test/%d.jpg' % int(data))
            plt.imshow(test_img)
            prediction = pre_train_model.predict(np.expand_dims(x, axis=0))[0]
            print(prediction)
            if prediction[0] < 0.5:
                plt.title('dog %.4f%%' % (100 - prediction[0]*100))
            else:
                plt.title('cat %.4f%%' % (prediction[0] * 100))

            plt.axis('off')
            plt.show()

        else:
            print(' please input the value between 1 and 2500')


    def data_augmentation(self):
        label = {0: 'Before Random-Erasing', 1: 'After Random-Erasing'}
        x = np.arange(0, 2)
        acc = np.array([0.9924, 0.9952])
        print('Before Random-Erasing: {}\n'.format(acc[0]))
        print('After Random-Erasing: {}\n'.format(acc[1]))
        plt.figure()
        plt.bar(x, acc * 100)
        plt.xticks(x, [label[0], label[1]])
        plt.grid(axis='y', linestyle='dotted', color='b')
        plt.ylabel('Accuracy(%)')
        plt.show()


    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.groupBox.setTitle(_translate("MainWindow", "5. Dogs and Cats classification Using ResNet50"))
        self.pushButton.setText(_translate("MainWindow", "1. Show Model Structure"))
        self.pushButton_2.setText(_translate("MainWindow", "2. Show TensorBoard"))
        self.pushButton_3.setText(_translate("MainWindow", "3. Test"))
        self.pushButton_4.setText(_translate("MainWindow", "4. Data Augmentatoin"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

