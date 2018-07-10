import os
from collections import OrderedDict
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from keras.layers import Layer, Input, Dense, Activation, Conv2D, GlobalAveragePooling2D
from keras.models import Model
from keras.datasets import cifar10
from keras import backend as K

from switchnorm import SwitchNormalization

nb_classes = 10
img_rows, img_cols = 32, 32
img_channels = 3

img_dim = (img_channels, img_rows, img_cols) if K.image_dim_ordering() == "th" else (img_rows, img_cols, img_channels)
bottleneck = False
width = 1
weight_decay = 1e-4


ip = Input(shape=img_dim)

x = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(ip)
x = SwitchNormalization(axis=-1, momentum=0.98)(x)
x = Activation('relu')(x)

x = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(x)
x = SwitchNormalization(axis=-1, momentum=0.98)(x)
x = Activation('relu')(x)

x = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal', strides=(2, 2))(x)
x = SwitchNormalization(axis=-1, momentum=0.98)(x)
x = Activation('relu')(x)

x = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal')(x)
x = SwitchNormalization(axis=-1, momentum=0.98)(x)
x = Activation('relu')(x)

x = Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal', strides=(2, 2))(x)
x = SwitchNormalization(axis=-1, momentum=0.98)(x)
x = Activation('relu')(x)

x = Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal')(x)
x = SwitchNormalization(axis=-1, momentum=0.98)(x)
x = Activation('relu')(x)

x = GlobalAveragePooling2D()(x)
x = Dense(nb_classes, activation='softmax')(x)

model = Model(ip, x)
print("Model created")

print("Model created")

model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])
print("Finished compiling")
print("Building model...")

(trainX, trainY), (testX, testY) = cifar10.load_data()

trainX = trainX.astype('float32')
testX = testX.astype('float32')

trainX = (trainX / 127.5) - 1.0
testX = (testX / 127.5) - 1.0

# Load model
weights_file = "weights/Baseline-CIFAR10.h5"
if os.path.exists(weights_file):
    model.load_weights(weights_file, by_name=True)
    print("Model loaded.")
else:
    print("Please train a CIFAR model first !")
    exit()

layer_dict = OrderedDict([(i, layer.name)
                          for i, layer in enumerate(model.layers)
                          if layer.__class__.__name__ == 'SwitchNormalization'])


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


fig, ax = plt.subplots(len(layer_dict), 2, squeeze=False, sharex=True, sharey=True)
fig.subplots_adjust(hspace=.5)

for i, (id, name) in enumerate(layer_dict.items()):
    print("Found %s at index %d" % (name, id))
    layer = model.layers[id]  # type: Layer

    weights = layer.get_weights()

    mean_weights = softmax(weights[2])
    variance_weights = softmax(weights[3])

    print("mean", mean_weights)
    print("variance", variance_weights)

    print()

    plt.sca(ax[i, 0])
    plt.barh(list(range(3)), mean_weights)
    plt.yticks(list(range(3)), ['instance', 'layer', 'batch'])
    plt.xlim([0.0, 1.0])
    plt.title(name + ' - mean')

    plt.sca(ax[i, 1])
    plt.barh(list(range(3)), variance_weights)
    plt.yticks(list(range(3)), ['instance', 'layer', 'batch'])
    plt.xlim([0.0, 1.0])
    plt.title(name + ' - var')

plt.show()
