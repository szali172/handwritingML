# This program decodes handwritten images of digits provided through the MNIST database
#    and displays the number using a neural network system

# Imports
import numpy as np
import mnist
import matplotlib.pyplot as plt  # Graph
from tensorflow.keras.models import Sequential  # Neural Network Architecture
from tensorflow.keras.layers import Dense  # Layers of Neural Network
from tensorflow.keras.utils import to_categorical  # Transforms our data

# Load and set the data
train_images = mnist.train_images()
trains_labels = mnist.train_labels()
test_images = mnist.test_images()
test_labels = mnist.test_labels()

# Normalize the pixel values from [0, 255] to [-0.5, 0.5]
train_images = (train_images / 255) - 0.5
test_images = (test_images / 255) - 0.5

# Flatten each 28x28 image into a 784 dimensional vector
train_images = train_images.reshape((-1, 784))  # 60,000 rows and 784 cols
test_images = test_images.reshape((-1, 784))  # 10,000 rows and 784 cols

# Model: 3 layers
# 2 layers with 64 neurons and the relu function
# 1 layer with 10 neurons and the softmax function
# DENSE method adds in the layers and neurons
model = Sequential()
model.add(Dense(64, activation='relu', input_dim=784))  # First layer
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))  # Final layer

# Compile the model using the LOSS function
# Loss function measures how well the model did in training,
#    then tries to improve using OPTIMIZER
model.compile(optimizer='adam',
              loss='categorical_crossentropy',  # allows us to use classes that are greater than two
              metrics=['accuracy']
              )

# Train the model
model.fit(train_images,
          to_categorical(trains_labels),  # transforms it into a ten dimensional vector
          epochs=5,  # number of iterations over the entire data set
          batch_size=32  # number of samples
          )

# Evaluates the Loss and Accuracy of the model
model.evaluate(test_images,
               to_categorical(test_labels)
               )

# predict the first 10 images
predictions = model.predict(test_images[:10])
print(np.argmax(predictions, axis=1))  # what the model is predicting
print(test_labels[:10])  # what the actual values are

# prints the images
for i in range(0, 10):
    first_image = test_images[i]
    first_image = np.array(first_image, dtype='float')
    pixels = first_image.reshape((28, 28))
    plt.imshow(pixels, cmap='gray')
    plt.show()

# Highest accuracy achieved = 96.84%
