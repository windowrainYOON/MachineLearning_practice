import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
import numpy as np
import logging
tf.get_logger().setLevel(logging.ERROR)

EPOCHS = 20 # Train EPOCHS iterations
BATCH_SIZE = 1
SEED_VAL = 7

# load training and test datasets
mnist = keras.datasets.mnist
(train_full_images, train_full_labels), (test_images, test_labels) = mnist.load_data()

train_images, train_labels = train_full_images[:-5000], train_full_labels[:-5000]
val_images, val_labels = train_full_images[-5000:], train_full_labels[-5000:]

# Standardize the data.
mean = np.mean(train_images)
stddev = np.std(train_images)
train_images = (train_images - mean) / stddev
val_images = (val_images - mean) / stddev
test_images = (test_images - mean) / stddev

# One-hot encode labels
train_labels = to_categorical(train_labels, num_classes=10)
val_labels = to_categorical(val_labels, num_classes=10)
test_labels = to_categorical(test_labels, num_classes=10)


# Object used to initialize weights
tf.random.set_seed(SEED_VAL)
initializer = keras.initializers.RandomUniform(
    minval=-0.1, maxval=0.1, seed=SEED_VAL
)

# Create a Sequential model.
# 784 inputs
# Two Dense (fully connected layers with 25 and 10 neurons.
# tanh as activation function for hidden layer.
# Logistic (sigmoid) as activation function for output layer.
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(25, activation='tanh', kernel_initializer=initializer, bias_initializer='zeros'),
    keras.layers.Dense(10, activation='sigmoid', kernel_initializer=initializer, bias_initializer='zeros')
])


# Use stochastic gradient descent (SGD) with learning rate of 0.01 and no other bells and whistles.
# MSE as loss function and report accuracy during traing.
opt = keras.optimizers.SGD(learning_rate=0.01)

model.compile(loss='mean_squared_error', optimizer=opt, metrics = ['accuracy'])

history = model.fit(train_images, train_labels, validation_data=(val_images, val_labels), epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=2, shuffle=True)

from matplotlib import pyplot

fig, loss_ax = pyplot.subplots()
acc_ax = loss_ax.twinx()

loss_ax.plot(history.history['loss'], 'y', label='train loss')
loss_ax.plot(history.history['val_loss'], 'r', label='validation loss')
loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
loss_ax.legend(loc='lower left')

acc_ax.plot(history.history['accuracy'], 'b', label='train accuracy')
acc_ax.plot(history.history['val_accuracy'], 'g', label='validation accuracy')
acc_ax.set_ylabel('accuracy')
acc_ax.legend(loc='upper left')

pyplot.show()