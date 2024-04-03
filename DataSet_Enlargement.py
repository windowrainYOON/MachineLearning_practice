import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Rescaling
from tensorflow.keras.layers import RandomFlip
from tensorflow.keras.layers import RandomRotation
from tensorflow.keras.layers import RandomContrast
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
model.add(RandomFlip(mode="horizontal", seed=42))
model.add(RandomRotation(factor=0.05, seed=42))
model.add(RandomContrast(factor=0.2, seed=42))
model.add(Conv2D(64, (4, 4), activation='relu', padding='same', input_shape=(32, 32, 3)))
model.add(Dropout(0.2))
model.add(Conv2D(64, (2, 2), activation='relu', padding='same', strides=(2,2)))
model.add(Dropout(0.2))
model.add(Conv2D(32, (3, 3), strides=(2, 2), activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(32, (3, 3), strides=(2, 2), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2,2), strides=2))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='softmax'))

# Compile and train the model.
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

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