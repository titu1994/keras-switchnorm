from __future__ import print_function

import os

import numpy as np
import sklearn.metrics as metrics

from keras.layers import Input, Dense, Activation, Conv2D, GlobalAveragePooling2D, BatchNormalization
from keras.models import Model
from keras.datasets import cifar10
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras import backend as K


if not os.path.exists('weights/'):
    os.makedirs('weights/')

batch_size = 200
nb_classes = 10
nb_epoch = 100

img_rows, img_cols = 32, 32
img_channels = 3

img_dim = (img_channels, img_rows, img_cols) if K.image_dim_ordering() == "th" else (img_rows, img_cols, img_channels)

ip = Input(shape=img_dim)

# 0.98 is stable, whereas 0.99 is unstable for this model.
# Stability here comes at the cost of slight performance.

x = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(ip)
x = BatchNormalization(axis=-1)(x)
x = Activation('relu')(x)

x = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(x)
x = BatchNormalization(axis=-1)(x)
x = Activation('relu')(x)

x = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal', strides=(2, 2))(x)
x = BatchNormalization(axis=-1)(x)
x = Activation('relu')(x)

x = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal')(x)
x = BatchNormalization(axis=-1)(x)
x = Activation('relu')(x)

x = Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal', strides=(2, 2))(x)
x = BatchNormalization(axis=-1)(x)
x = Activation('relu')(x)

x = Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal')(x)
x = BatchNormalization(axis=-1)(x)
x = Activation('relu')(x)

x = GlobalAveragePooling2D()(x)
x = Dense(nb_classes, activation='softmax')(x)

model = Model(ip, x)
print("Model created")

model.summary()
optimizer = Adam(1e-3)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=["accuracy"])
print("Finished compiling")
print("Building model...")

(trainX, trainY), (testX, testY) = cifar10.load_data()

trainX = trainX.astype('float32')
testX = testX.astype('float32')

trainX = (trainX / 127.5) - 1.0
testX = (testX / 127.5) - 1.0

Y_train = np_utils.to_categorical(trainY, nb_classes)
Y_test = np_utils.to_categorical(testY, nb_classes)

generator = ImageDataGenerator(rotation_range=10,
                               width_shift_range=5. / 32,
                               height_shift_range=5. / 32,
                               horizontal_flip=False)

generator.fit(trainX, seed=0)

# Load model
weights_file = "weights/Baseline-BatchNorm-CIFAR10.h5"
if os.path.exists(weights_file):
    model.load_weights(weights_file, by_name=True)
    print("Model loaded.")

out_dir = "weights/"

lr_reducer = ReduceLROnPlateau(monitor='val_acc', factor=np.sqrt(0.5),
                               cooldown=0, patience=5, min_lr=1e-4, verbose=1)
model_checkpoint = ModelCheckpoint(weights_file, monitor="val_acc", save_best_only=True,
                                   save_weights_only=True, verbose=1)

callbacks = [lr_reducer, model_checkpoint]

model.fit_generator(generator.flow(trainX, Y_train, batch_size=batch_size),
                    steps_per_epoch=len(trainX) // batch_size, epochs=nb_epoch,
                    callbacks=callbacks,
                    validation_data=(testX, Y_test),
                    validation_steps=testX.shape[0] // batch_size, verbose=1)

yPreds = model.predict(testX)
yPred = np.argmax(yPreds, axis=1)
yTrue = testY

accuracy = metrics.accuracy_score(yTrue, yPred) * 100
error = 100 - accuracy
print("Accuracy : ", accuracy)
print("Error : ", error)
