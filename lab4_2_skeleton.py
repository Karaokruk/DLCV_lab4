from __future__ import print_function

import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping

import matplotlib.pyplot as plt
import numpy as np


print('tensorflow:', tf.__version__)
print('keras:', tensorflow.keras.__version__)


##Uncomment the following two lines if you get CUDNN_STATUS_INTERNAL_ERROR initialization errors.
## (it happens on RTX 2060 on room 104/moneo or room 204/lautrec)
#physical_devices = tf.config.experimental.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(physical_devices[0], True)


#load (first download if necessary) the CIFAR10 dataset
# data is already split in train and test datasets
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

a = x_train.shape[0]
b = x_train.shape[1]
c = x_train.shape[2]
d = x_train.shape[3]
print("a={}, b={}, c={}, d={}".format(a, b, c, d))

## Display one image and corresponding label
import matplotlib
import matplotlib.pyplot as plt
#i = np.random.randint(a)
#plt.imshow(x_train[i], cmap = matplotlib.cm.binary)
#plt.axis("off")
#plt.show()
#Let start our work: creating a convolutional neural network


num_classes = 10
def one_hot_encode(y, digits):
    print( "shape of y: ",y.shape)
    examples = y.shape[0]
    y = y.reshape(1, examples)
    Y_new = np.eye(digits)[y.astype('int32')]  #shape (1, 70000, 10)
    Y_new = Y_new.T.reshape(digits, examples)
    print( "shape of encoded y: ",Y_new.shape)
    Y_new = Y_new.T
    return Y_new

y_train = one_hot_encode(y_train, num_classes)
y_test = one_hot_encode(y_test, num_classes)
print(x_train.shape)
print(y_train.shape)

# NE PAS OUBLIER DE ONE HOT ENCODED XTRAIN ET YTARIN LOOOOL (peut-être aussi x_test & y_test)

def cnn():
    model = Sequential()
    model.add(Conv2D(32, kernel_initializer="normal", kernel_size=(3,3), activation='relu', input_shape=(32, 32, 3)))
    #model.add(Conv2D(128, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

nb_epochs = 1
batch_size = 64

model = cnn()
model.summary()
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)
hist = model.fit(x_train,
                 y_train,
                 validation_data=(x_test, y_test),
                 epochs=nb_epochs,
                 batch_size=batch_size,
                 callbacks=[callback])

#plt.plot(hist.history['accuracy'], list(range(hist.history['accuracy'])), label='accuracy')
#plt.plot(hist.history['val_accuracy'])
#plt.title('model accuracy')
#plt.xlabel('epoch')
#plt.ylabel('accuracy')
#plt.legend(['train','test'],loc='upper left')
#plt.show()
