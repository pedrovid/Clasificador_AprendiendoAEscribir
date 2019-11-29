from emnist import extract_training_samples
from emnist import extract_test_samples
import numpy as np
from tensorflow import keras
from keras.utils import to_categorical

images, labels = extract_training_samples('letters')
imagestest, labelstest = extract_test_samples('letters')
clase=["","a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z"]

# creacion del modelo que clasificara las letras
print(labels.shape)
labels =to_categorical(labels)
images=images.astype('float32')
images/=255

labelstest = to_categorical(labelstest)
imagestest=imagestest.astype('float32')
imagestest/=255
images=np.vstack((images,imagestest))
images=np.asarray(images)

labels=np.vstack((labels,labelstest))
images = images.reshape(images.shape[0], 28, 28, 1).astype('float32')



model = keras.models.Sequential()
model.add(keras.layers.Conv2D(32, (3, 3), input_shape=(28, 28, 1), activation='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dense(27, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(images, labels, epochs=20, batch_size=200, verbose=2)
model.save('../Weights/my_model.h5')
