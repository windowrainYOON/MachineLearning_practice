import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
import numpy as np
from matplotlib import pyplot

mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

print('dimensions of train_images: ', train_images.shape)
print('dimensions of train_labels: ', train_labels.shape)
print('dimensions of test_images: ', train_images.shape)
print('dimensions of test_images: ', test_labels.shape)

pyplot.figure(figsize=(3,3))
image = np.reshape(train_images[1], [28, 28])
pyplot.imshow(image, cmap='Greys')
pyplot.show()