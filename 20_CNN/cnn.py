from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense


# init graphic card
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto(allow_soft_placement=False)
config.gpu_options.allow_growth = True
# config.log_device_placement = True
sess = tf.Session(config=config)
set_session(sess)

# creating the cnn

classifier = Sequential()

# first cnn layer
classifier.add(
    Convolution2D(filters=32, kernel_size=(3, 3), padding='valid', input_shape=(64, 64, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# second cnn layer
classifier.add(
    Convolution2D(filters=32, kernel_size=(3, 3), padding='valid', activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))


classifier.add(Flatten())

# normal ann layer
classifier.add(Dense(128, use_bias=True, kernel_initializer='random_uniform', activation='relu'))
classifier.add(Dense(1, use_bias=True, kernel_initializer='random_uniform', activation='sigmoid'))
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

training_set = train_datagen.flow_from_directory(
    'dataset/training_set',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary')

test_set = test_datagen.flow_from_directory(
    'dataset/test_set',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary')

classifier.fit_generator(
    training_set,
    use_multiprocessing=False,
    verbose=1,
    workers=5,
    steps_per_epoch=8000,
    epochs=2,
    validation_data=test_set,
    validation_steps=2000)
