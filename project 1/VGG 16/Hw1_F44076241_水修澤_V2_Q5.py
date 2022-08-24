# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Hw1_F44076241_水修澤_V2_Q5.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Conv2D, MaxPool2D, Flatten, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import cifar10
from keras.utils import np_utils
from tensorflow.keras.optimizers import SGD
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random



class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(190, 80, 391, 401))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.groupBox.setFont(font)
        self.groupBox.setObjectName("groupBox")
        self.pushButton = QtWidgets.QPushButton(self.groupBox)
        self.pushButton.setGeometry(QtCore.QRect(90, 50, 211, 28))
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_2.setGeometry(QtCore.QRect(90, 110, 211, 28))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_3 = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_3.setGeometry(QtCore.QRect(90, 170, 211, 28))
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_4 = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_4.setGeometry(QtCore.QRect(90, 230, 211, 28))
        self.pushButton_4.setObjectName("pushButton_4")
        self.pushButton_5 = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_5.setGeometry(QtCore.QRect(90, 340, 211, 28))
        self.pushButton_5.setObjectName("pushButton_5")
        self.lineEdit = QtWidgets.QLineEdit(self.groupBox)
        self.lineEdit.setGeometry(QtCore.QRect(90, 290, 211, 22))
        self.lineEdit.setObjectName("lineEdit")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 25))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.pushButton.clicked.connect(self.show_train_image)
        self.pushButton_2.clicked.connect(self.show_hyperparameters)
        self.pushButton_3.clicked.connect(self.show_model)
        self.pushButton_4.clicked.connect(self.show_accuracy)
        self.pushButton_5.clicked.connect(self.test)
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def show_train_image(self):
        label_dict = {0: "airplane", 1: "automobile", 2: "bird", 3: "cat", 4: "deer", 5: "dog", 6: "frog", 7: "horse",
                      8: "ship", 9: "truck"}

        def plot_cifar10(images, labels):
            plt.figure(figsize=(5, 5))
            for i in range(0, 9):
                random_num = random.randint(0, 50000)
                plt.subplot(3, 3, i + 1)
                plt.imshow(images[random_num])
                title = label_dict[labels[random_num][0]]
                plt.title(title, fontsize=10)
                plt.xticks([])
                plt.yticks([])
            plt.show()

        (x_train_image, y_train_label), (x_test_image, y_test_label) = cifar10.load_data()
        plot_cifar10(x_train_image, y_train_label)

    def show_hyperparameters(self):
        lr = 0.001
        batch_size = 32
        print('hyperparameters: ')
        print('batch_size: {}'.format(batch_size))
        print('learning rate: {}'.format(lr))
        print('optimizer: SGD')

    def show_model(self):
        # create VGG16 model
        model = Sequential()

        model.add(Conv2D(input_shape=(32, 32, 3), filters=64, kernel_size=(3, 3), padding="same", name='Conv2d-1'))
        model.add(BatchNormalization(name='BatchNorm-1'))
        model.add(Activation("relu", name='ReLU-1'))

        model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same", name='Conv2d-2'))
        model.add(BatchNormalization(name='BatchNorm-2'))
        model.add(Activation("relu", name='ReLU-2'))

        model.add(MaxPool2D(pool_size=(2, 2), name='MaxPool-3'))

        model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", name='Conv2d-4'))
        model.add(BatchNormalization(name='BatchNorm-4'))
        model.add(Activation("relu", name='ReLU-4'))

        model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", name='Conv2d-5'))
        model.add(BatchNormalization(name='BatchNorm-5'))
        model.add(Activation("relu", name='ReLU-5'))

        model.add(MaxPool2D(pool_size=(2, 2), name='MaxPool-6'))

        model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", name='Conv2d-7'))
        model.add(BatchNormalization(name='BatchNorm-7'))
        model.add(Activation("relu", name='ReLU-7'))

        model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", name='Conv2d-8'))
        model.add(BatchNormalization(name='BatchNorm-8'))
        model.add(Activation("relu", name='ReLU-8'))

        model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", name='Conv2d-9'))
        model.add(BatchNormalization(name='BatchNorm-9'))
        model.add(Activation("relu", name='ReLU-9'))

        model.add(MaxPool2D(pool_size=(2, 2), name='MaxPool-10'))

        model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", name='Conv2d-11'))
        model.add(BatchNormalization(name='BatchNorm-11'))
        model.add(Activation("relu", name='ReLU-11'))

        model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", name='Conv2d-12'))
        model.add(BatchNormalization(name='BatchNorm-12'))
        model.add(Activation("relu", name='ReLU-12'))

        model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", name='Conv2d-13'))
        model.add(BatchNormalization(name='BatchNorm-13'))
        model.add(Activation("relu", name='ReLU-13'))

        model.add(MaxPool2D(pool_size=(2, 2), name='MaxPool-14'))

        model.add(Flatten(name='Flatten'))
        model.add(Dense(units=4096, activation="relu", name='fc-15'))
        model.add(BatchNormalization(name='BatchNorm-15'))
        model.add(Dropout(0.5, name='Dropout-15'))

        model.add(Dense(units=4096, activation="relu", name='fc-16'))
        model.add(BatchNormalization(name='BatchNorm-16'))
        model.add(Dropout(0.5, name='Dropout-16'))

        model.add(Dense(10, activation="softmax", name='prediction-17'))

        model.summary()

    def train_model(self):
        # create VGG16 model
        model = Sequential()

        model.add(Conv2D(input_shape=(32, 32, 3), filters=64, kernel_size=(3, 3), padding="same", name='Conv2d-1'))
        model.add(BatchNormalization(name='BatchNorm-1'))
        model.add(Activation("relu", name='ReLU-1'))

        model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same", name='Conv2d-2'))
        model.add(BatchNormalization(name='BatchNorm-2'))
        model.add(Activation("relu", name='ReLU-2'))

        model.add(MaxPool2D(pool_size=(2, 2), name='MaxPool-3'))

        model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", name='Conv2d-4'))
        model.add(BatchNormalization(name='BatchNorm-4'))
        model.add(Activation("relu", name='ReLU-4'))

        model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", name='Conv2d-5'))
        model.add(BatchNormalization(name='BatchNorm-5'))
        model.add(Activation("relu", name='ReLU-5'))

        model.add(MaxPool2D(pool_size=(2, 2), name='MaxPool-6'))

        model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", name='Conv2d-7'))
        model.add(BatchNormalization(name='BatchNorm-7'))
        model.add(Activation("relu", name='ReLU-7'))

        model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", name='Conv2d-8'))
        model.add(BatchNormalization(name='BatchNorm-8'))
        model.add(Activation("relu", name='ReLU-8'))

        model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", name='Conv2d-9'))
        model.add(BatchNormalization(name='BatchNorm-9'))
        model.add(Activation("relu", name='ReLU-9'))

        model.add(MaxPool2D(pool_size=(2, 2), name='MaxPool-10'))

        model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", name='Conv2d-11'))
        model.add(BatchNormalization(name='BatchNorm-11'))
        model.add(Activation("relu", name='ReLU-11'))

        model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", name='Conv2d-12'))
        model.add(BatchNormalization(name='BatchNorm-12'))
        model.add(Activation("relu", name='ReLU-12'))

        model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", name='Conv2d-13'))
        model.add(BatchNormalization(name='BatchNorm-13'))
        model.add(Activation("relu", name='ReLU-13'))

        model.add(MaxPool2D(pool_size=(2, 2), name='MaxPool-14'))

        model.add(Flatten(name='Flatten'))
        model.add(Dense(units=4096, activation="relu", name='fc-15'))
        model.add(BatchNormalization(name='BatchNorm-15'))
        model.add(Dropout(0.5, name='Dropout-15'))

        model.add(Dense(units=4096, activation="relu", name='fc-16'))
        model.add(BatchNormalization(name='BatchNorm-16'))
        model.add(Dropout(0.5, name='Dropout-16'))

        model.add(Dense(10, activation="softmax", name='prediction-17'))

        model.summary()

        lr = 0.001
        nb_epoch = 50
        decay = lr / nb_epoch
        momentum = 0.9
        batch_size = 32
        opt = SGD(learning_rate=lr, decay=decay, momentum=momentum, nesterov=True)
        print('hyperparameters: ')
        print('batch_size: {}'.format(batch_size))
        print('learning rate: {}'.format(lr))
        print('optimizer: SGD')

        # model.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
        # hist = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=25, batch_size=batch_size, verbose=1)
        loss, accuracy = model.evaluate(x_test, y_test)
        print('test: ')
        print('loss: {}'.format(loss))
        print('accuracy: {}'.format(accuracy))

        fig1 = plt.figure()
        plt.plot(hist.history['accuracy'], label='training accuracy')
        plt.plot(hist.history['val_accuracy'], label='testing accuracy')
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(loc='lower right')
        fig1.savefig('Q5_VGG16_accuracy.png')

        fig2 = plt.figure()
        plt.plot(hist.history['loss'], label='training loss')
        plt.plot(hist.history['val_loss'], label='testing loss')
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(loc='upper right')
        fig2.savefig('Q5_VGG16_loss.png')

        model.save('VGG16_cifar10.h5')

    def show_accuracy(self):
        VGG16_accuracy = cv2.imread('Q5_VGG16_accuracy.png')
        VGG16_loss = cv2.imread('Q5_VGG16_loss.png')
        screenshot = cv2.imread('screenshot_loss_accuracy.png')
        cv2.imshow('Accuracy', VGG16_accuracy)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imshow('Loss', VGG16_loss)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imshow('Screenshot', screenshot)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def test(self):
        label_dict = {0: "plane", 1: "mobile", 2: "bird", 3: "cat", 4: "deer", 5: "dog", 6: "frog", 7: "horse",
                      8: "ship", 9: "truck"}
        (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
        x_test = X_test.astype('float32') / 255
        data = self.lineEdit.text()
        pre_train_model = keras.models.load_model('VGG16_cifar10.h5')
        if (int(data) - 1 > -1 and int(data) - 1 < 10000):
            test_img = plt.subplot(2, 1, 1)
            test_img.imshow(X_test[int(data) - 1])
            feature = pre_train_model.predict(x_test[int(data) - 1].reshape(-1, 32, 32, 3))
            print(feature)
            x = np.arange(0, 10)
            plt.subplot(2, 1, 2)
            plt.bar(x, feature.reshape(10))
            plt.xticks(x, [label_dict[0], label_dict[1], label_dict[2], label_dict[3], label_dict[4], label_dict[5], label_dict[6], label_dict[7], label_dict[8], label_dict[9]])
            plt.show()
        else:
            print(' please input the value between 1 and 10000')





    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.groupBox.setTitle(_translate("MainWindow", "VGG16 TEST"))
        self.pushButton.setText(_translate("MainWindow", "1. Show Train Images"))
        self.pushButton_2.setText(_translate("MainWindow", "2. Show HyperParameters"))
        self.pushButton_3.setText(_translate("MainWindow", "3. Show Model Shortcut"))
        self.pushButton_4.setText(_translate("MainWindow", "4. Show Accuracy"))
        self.pushButton_5.setText(_translate("MainWindow", "5.Test"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

