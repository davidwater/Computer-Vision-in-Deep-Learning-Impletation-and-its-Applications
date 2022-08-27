import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import cifar10
import random


label_dict={0:"airplane", 1:"automobile", 2:"bird", 3:"cat", 4:"deer", 5:"dog", 6:"frog", 7:"horse", 8:"ship", 9:"truck"}

def plot_cifar10(images, labels):
    plt.figure(figsize=(5, 5))
    for i in range(0, 9):
        random_num = random.randint(0, 50000)
        plt.subplot(3, 3, i+1)
        plt.imshow(images[random_num])
        title = label_dict[labels[random_num][0]]
        plt.title(title, fontsize = 10)
        plt.xticks([])
        plt.yticks([])
    plt.show()

(x_train_image, y_train_label), (x_test_image, y_test_label) = cifar10.load_data()
plot_cifar10(x_train_image, y_train_label)
