import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Conv2D, MaxPool2D, Flatten, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import cifar10
from keras.utils import np_utils
from tensorflow.keras.optimizers import SGD
import numpy as np
import matplotlib.pyplot as plt

(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
x_train = X_train.astype('float32')/255
x_test = X_test.astype('float32')/255
y_train = np_utils.to_categorical(Y_train,10)
y_test = np_utils.to_categorical(Y_test,10)

# create VGG16 model
model = Sequential()

model.add(Conv2D(input_shape=(32,32,3),filters=64,kernel_size=(3,3),padding="same", name='Conv2d-1'))
model.add(BatchNormalization(name='BatchNorm-1'))
model.add(Activation("relu", name='ReLU-1'))

model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", name='Conv2d-2'))
model.add(BatchNormalization(name='BatchNorm-2'))
model.add(Activation("relu", name='ReLU-2'))

model.add(MaxPool2D(pool_size=(2,2), name='MaxPool-3'))

model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", name='Conv2d-4'))
model.add(BatchNormalization(name='BatchNorm-4'))
model.add(Activation("relu", name='ReLU-4'))

model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", name='Conv2d-5'))
model.add(BatchNormalization(name='BatchNorm-5'))
model.add(Activation("relu", name='ReLU-5'))

model.add(MaxPool2D(pool_size=(2,2), name='MaxPool-6'))

model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", name='Conv2d-7'))
model.add(BatchNormalization(name='BatchNorm-7'))
model.add(Activation("relu", name='ReLU-7'))


model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", name='Conv2d-8'))
model.add(BatchNormalization(name='BatchNorm-8'))
model.add(Activation("relu", name='ReLU-8'))

model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", name='Conv2d-9'))
model.add(BatchNormalization(name='BatchNorm-9'))
model.add(Activation("relu", name='ReLU-9'))

model.add(MaxPool2D(pool_size=(2,2), name='MaxPool-10'))

model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", name='Conv2d-11'))
model.add(BatchNormalization(name='BatchNorm-11'))
model.add(Activation("relu", name='ReLU-11'))

model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", name='Conv2d-12'))
model.add(BatchNormalization(name='BatchNorm-12'))
model.add(Activation("relu", name='ReLU-12'))

model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", name='Conv2d-13'))
model.add(BatchNormalization(name='BatchNorm-13'))
model.add(Activation("relu", name='ReLU-13'))

model.add(MaxPool2D(pool_size=(2,2), name='MaxPool-14'))

model.add(Flatten(name='Flatten'))
model.add(Dense(units=4096, activation="relu", name='fc-15'))
model.add(BatchNormalization(name='BatchNorm-15'))
model.add(Dropout(0.5, name='Dropout-15'))

model.add(Dense(units=4096,activation="relu", name='fc-16'))
model.add(BatchNormalization(name='BatchNorm-16'))
model.add(Dropout(0.5, name='Dropout-16'))

model.add(Dense(10, activation="softmax", name='prediction-17'))

model.summary()

lr = 0.001
nb_epoch = 50
decay = lr/nb_epoch
momentum = 0.9
batch_size = 32
opt = SGD(learning_rate=lr, decay=decay, momentum=momentum, nesterov=True)
print('hyperparameters: ')
print('batch_size: {}'.format(batch_size))
print('learning rate: {}'.format(lr))
print('optimizer: SGD')

model.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
hist = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=20, batch_size=batch_size, verbose=1)
loss, accuracy = model.evaluate(x_test, y_test)
print('test: ')
print('loss: {}'.format(loss))
print('accuracy: {}'.format(accuracy))

fig1 = plt.figure()
plt.plot(hist.history['accuracy'], label = 'training accuracy')
plt.plot(hist.history['val_accuracy'], label = 'testing accuracy')
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(loc = 'lower right')
fig1.savefig('Q5_VGG16_accuracy.png')

fig2 = plt.figure()
plt.plot(hist.history['loss'], label = 'training loss')
plt.plot(hist.history['val_loss'], label = 'testing loss')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc = 'upper right')
fig2.savefig('Q5_VGG16_loss.png')

model.save('VGG16_cifar10.h5')







