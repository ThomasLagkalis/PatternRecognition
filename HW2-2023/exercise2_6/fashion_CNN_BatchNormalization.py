from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

from plots import plot_some_data, plot_some_predictions


fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Scale these values to a range of 0 to 1 before feeding them to the neural network model. 
# To do so, divide the values by 255. 
# It's important that the training set and the testing set be preprocessed in the same way
train_images = train_images / 255.0

test_images = test_images / 255.0

train_images_reshaped = np.reshape(train_images, (60000, 28, 28, 1)) # reshape to mum_train_images X height X width X channels, where channels = 1
test_images_reshaped = np.reshape(test_images, (10000, 28, 28, 1)) # reshape


# Build the model
# Building the neural network requires configuring the layers of the model, then compiling the model.

model = keras.Sequential([

    keras.layers.Conv2D(filters=32, strides=1,kernel_size=(3,3), padding="same", input_shape=(28,28,1)), #Conv1
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),
    keras.layers.Conv2D(filters=32, strides=1,kernel_size=(3,3), padding="same", input_shape=(28,28,1)), #conv2
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),
    keras.layers.MaxPool2D(pool_size=(2,2), padding="valid"), #Max_Pool1
    keras.layers.Conv2D(filters=64, strides=1, padding="same", kernel_size=(3,3), input_shape=(14,14,32)), #Conv3
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),
    keras.layers.Conv2D(filters=64, strides=1, padding="same", kernel_size=(3,3), input_shape=(14,14,64)), #Conv4
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),
    keras.layers.MaxPool2D(pool_size=(2,2), padding="valid"), #MaxPool2
    keras.layers.Conv2D(filters=128, strides=1, padding="same", kernel_size=(3,3), input_shape=(7,7,64)), #Conv5
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),
    keras.layers.MaxPool2D(pool_size=(2,2), padding="valid"), #MaxPool3
    keras.layers.Flatten(input_shape=(3,3,128)), #Flat Layer
    keras.layers.Dense(units=1152, activation='relu'), #Dense1
    keras.layers.BatchNormalization(),
    keras.layers.Dense(units=200, activation='relu'), #Dense2
    keras.layers.Dense(units=10, activation="softmax")
]) # fill the model

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

#Train the model
# Training the neural network model requires the following steps:

#   1. Feed the training data to the model. In this example, the training data is in the train_images and train_labels arrays.
#   2. The model learns to associate images and labels.
#   3. You ask the model to make predictions about a test set—in this example, the test_images array.
#   4. Verify that the predictions match the labels from the test_labels array.

model.fit(train_images_reshaped, train_labels, epochs=50, validation_data=(test_images_reshaped, test_labels))

# Evaluate accuracy
test_loss, test_acc = model.evaluate(test_images_reshaped,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc)

# Make predictions
# With the model trained, you can use it to make predictions about some images. 
# The model's linear outputs, logits. 
# Attach a softmax layer to convert the logits to probabilities, which are easier to interpret. 
probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])

predictions = probability_model.predict(test_images_reshaped)

plot_some_predictions(test_images, test_labels, predictions, class_names, num_rows=5, num_cols=3)


