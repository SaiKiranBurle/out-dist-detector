"""
Adopted from
https://github.com/titu1994/DenseNet/blob/master/cifar10.py
"""
import os.path

import numpy as np
import sklearn.metrics as metrics
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.datasets import cifar10
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils

import densenet

batch_size = 64
nb_classes = 10
nb_epoch = 300

dropout_rate = 0.0
OUT_DIR = "weights/"

model = densenet.DenseNet((32, 32, 3), depth=100,growth_rate=12, bottleneck=True, weights=None)
print "Model created"

model.summary()
optimizer = Adam(lr=1e-3)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=["accuracy"])
print "Finished compiling"
print "Building model..."

(trainX, trainY), (testX, testY) = cifar10.load_data()

trainX = trainX.astype('float32')
testX = testX.astype('float32')

trainX = densenet.preprocess_input(trainX)
testX = densenet.preprocess_input(testX)

Y_train = np_utils.to_categorical(trainY, nb_classes)
Y_test = np_utils.to_categorical(testY, nb_classes)

generator = ImageDataGenerator(rotation_range=15,
                               width_shift_range=5./32,
                               height_shift_range=5./32,
                               horizontal_flip=True)

generator.fit(trainX, seed=0)

# Load model
weights_file = "weights/DenseNet-100-12-CIFAR10.h5"
if os.path.exists(weights_file):
    # model.load_weights(weights_file, by_name=True)
    print("Model loaded.")

lr_reducer = ReduceLROnPlateau(monitor='val_acc', factor=np.sqrt(0.1),
                               cooldown=0, patience=5, min_lr=1e-5)
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
