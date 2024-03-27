import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Rescaling
import numpy as np
import logging
tf.get_logger().setLevel(logging.ERROR)

EPOCHS = 128
BATCH_SIZE = 32

# Load Dataset
cifar_dataset = keras.datasets.cifar10
(train_full_images, train_full_labels), (test_images, test_labels) = cifar_dataset.load_data()

train_images, train_labels = train_full_images[:-5000], train_full_labels[:-5000]
val_images, val_labels = train_full_images[-5000:], train_full_labels[-5000:]

# Model with two convolutional and one fully connected layer.
model = Sequential()
model.add(Rescaling(1./255))
model.add(Conv2D(64, (5, 5), strides=(2, 2),
                 activation='relu', padding='same',
                 input_shape=(32, 32, 3),
                 kernel_initializer='he_normal',
                 bias_initializer='zeros'))
model.add(Conv2D(64, (3, 3), strides=(2, 2),
                 activation='relu', padding='same',
                 kernel_initializer='he_normal',
                 bias_initializer='zeros'))
model.add(Flatten())
model.add(Dense(10, activation='softmax',
                kernel_initializer='glorot_uniform',
                bias_initializer='zeros'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Model structure
input_shape = (None, 32, 32, 3)
model.build(input_shape)
model.summary()

# Training the NN
history = model.fit(
    train_images, train_labels, validation_data =
    (val_images, val_labels), epochs = EPOCHS,
    batch_size = BATCH_SIZE, verbose = 2, shuffle=True
)

# Evaluating the trained model
from matplotlib import pyplot

pyplot.plot(history.history['accuracy'], 'b')
pyplot.plot(history.history['val_accuracy'], 'g')
pyplot.ylim(0.4, 1)
pyplot.xlabel('Epoch')
pyplot.ylabel('Accuracy')
pyplot.legend(['train', 'val'])
pyplot.show()

# Evaluating the trained model on the test set
model.evaluate(test_images, test_labels)