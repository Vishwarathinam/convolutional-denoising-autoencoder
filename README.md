### Convolutional Autoencoder for Image Denoising

### AIM:

To develop a convolutional autoencoder for image denoising application.

### Problem Statement and Dataset:
Image Denoising is the process of removing noise from the Images. The noise present in the images may be caused by various intrinsic or extrinsic conditions which are practically hard to deal with. The problem of Image Denoising is a very fundamental challenge in the domain of Image processing and Computer vision. Therefore, it plays an important role in a wide variety of domains where getting the original image is really important for robust performance.

Modeling image data requires a special approach in the neural network world. The best-known neural network for modeling image data is the Convolutional Neural Network (CNN). It can better retain the connected information between the pixels of an image. The particular design of the layers in a CNN makes it a better choice to process image data.

The CNN design can be used for image recognition/classification, or be used for image noise reduction or coloring. We can train the CNN model by taking many image samples as the inputs and labels as the outputs. We then use this trained CNN model to a new image to recognize if it is a “dog”, or “cat”, etc. CNN also can be used as an autoencoder for image noise reduction or coloring.

This program demonstrates how to implement a deep convolutional autoencoder for image denoising, mapping noisy digits images from the MNIST dataset to clean digits images. The MNIST database (Modified National Institute of Standards and Technology database) is a large database of handwritten digits that is commonly used for training various image processing systems. The training dataset in Keras has 60,000 records and the test dataset has 10,000 records. Each record has 28 x 28 pixels.

### DATASET:



### Convolution Autoencoder Network Model:
![n2](https://user-images.githubusercontent.com/95266350/203508928-6adfc450-8ba0-4086-9969-1f5f37e29134.png)





### DESIGN STEPS:

### STEP 1:
Import the necessary libraries and download the mnist dataset.

### STEP 2:
Load the dataset and scale the values for easier computation.

### STEP 3:
Add noise to the images randomly for the process of denoising it with the convolutional denoising autoencoders for both the training and testing sets.

### STEP 4:
Build the Neural Model for convolutional denoising autoencoders using Convolutional, Pooling and Up Sampling layers. Make sure the input shape and output shape of the model are identical.

### STEP 5:
Pass test data for validating manually. Compile and fit the created model.

### STEP 6:
Plot the Original, Noisy and Reconstructed Image predictions for visualization.

### STEP 7:
End the program


### PROGRAM:
~~~
### Developed by : Vishwa Rathinam. S,
### Reg.No : 212221240063,
Program to develop a convolutional autoencoder for image denoising application.

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import utils
from tensorflow.keras import models
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

(x_train, _), (x_test, _) = mnist.load_data()

x_train.shape

x_train_scaled = x_train.astype('float32') / 255.
x_test_scaled = x_test.astype('float32') / 255.
x_train_scaled = np.reshape(x_train_scaled, (len(x_train_scaled), 28, 28, 1))
x_test_scaled = np.reshape(x_test_scaled, (len(x_test_scaled), 28, 28, 1))

noise_factor = 0.5
x_train_noisy = x_train_scaled + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train_scaled.shape) 
x_test_noisy = x_test_scaled + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test_scaled.shape) 

x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

n = 10
plt.figure(figsize=(20, 2))
for i in range(1, n + 1):
    ax = plt.subplot(1, n, i)
    plt.imshow(x_test_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

input_img = keras.Input(shape=(28, 28, 1))

x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

autoencoder.summary()

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

autoencoder.fit(x_train_noisy, x_train_scaled,
                epochs=2,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test_noisy, x_test_scaled))
                
decoded_imgs = autoencoder.predict(x_test_noisy)

n = 10
plt.figure(figsize=(20, 4))
for i in range(1, n + 1):
    # Display original
    ax = plt.subplot(3, n, i)
    plt.imshow(x_test_scaled[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display noisy
    ax = plt.subplot(3, n, i+n)
    plt.imshow(x_test_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)    

    # Display reconstruction
    ax = plt.subplot(3, n, i + 2*n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
~~~

### OUTPUT:

### ADDING NOISE TO THE MNIST DATASET:


### AUTOENCODER.SUMMARY():
![n5](https://user-images.githubusercontent.com/95266350/203509144-cb5af41c-3aa9-4f07-9961-473a6d478a3e.png)

 
### Original vs Noisy Vs Reconstructed Image:
![n6](https://user-images.githubusercontent.com/95266350/203509410-252b3c9d-70d9-46cd-a28c-722b9b2cb1be.png)




### RESULT:
Thus, the program to develop a convolutional autoencoder for image denoising application is developed and executted successfully.


