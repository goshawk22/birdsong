from tensorflow.keras.layers import Input, Dense, Reshape, Dropout, Activation, MaxPooling2D, Flatten
from tensorflow.keras.layers import UpSampling2D, Conv2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import numpy as np

import random

def shuffle(data, labels):
    numbers = list(np.arange(len(data)))
    x = []
    y = []
    for _ in range(len(data)):
        choice = random.choice(numbers)
        x.append(data[choice])
        y.append(labels[choice])
    return np.array(x), np.array(y)
    
with np.load("largeData.npz") as data:
    X_train = data['X_train']
    Y_train = data['Y_train']
    englishName = data['englishName']

X_train, Y_train = shuffle(X_train, Y_train)


rows = 128
cols = 54
classes = 89
channels = 1
img_shape = (rows, cols, channels)

batch_size = 16
epochs = 8000

X_train = X_train.astype("float32") / 255
#x_test = x_test.astype("float32") / 255

for x in range(Y_train.shape[0]):
    Y_train[x] = int(Y_train[x]) - 1

Y_train = to_categorical(Y_train)

X_train = np.expand_dims(X_train, -1)

model = Sequential(
    [
        Input(shape=img_shape),
        Conv2D(32, kernel_size=(3, 3), activation="relu"),
        MaxPooling2D(),
        Conv2D(64, kernel_size=(3, 3), activation="relu"),
        MaxPooling2D(),
        Conv2D(128, kernel_size=(3, 3), activation="relu"),
        MaxPooling2D(),
        Flatten(),
        Dense(1024, activation='relu'),
        Dropout(0.5),
        Dense(1024, activation='relu'),
        Dropout(0.5),
        Dense(classes, activation="softmax"),
    ]
)

model.summary()

callbacks = [
    EarlyStopping(monitor='acc', patience=5, mode='max'),
    ModelCheckpoint(filepath='model.{epoch:02d}-{val_acc:.2f}.h5', period=10),
]

model.compile(loss='categorical_crossentropy',
    optimizer="adam",
    metrics=['accuracy'])


model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=callbacks)