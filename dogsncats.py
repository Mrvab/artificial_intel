#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 18:51:38 2020

@author: sadrobin
"""
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix

#from keras.preprocessing.image import load_img
#import matplotlib.pyplot as plt
import pickle

classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Dropout(.5))
# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Dropout(.5))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# Part 2 - Fitting the CNN to the images



train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

#print(training_set[0][0][0])
#img = training_set[0][0][0].reshape((64,64,3))
#plt.imshow(img)
#plt.title("none")
#plt.show()

# start training
#from keras.callbacks import ModelCheckpoint

#checkpoint = ModelCheckpoint(filepath = 'dogsandcat.model',save_best_only = True,verbose=1)

#history = model.fit(x_train,y_train,batch_size=32, epochs = 100,
#          validation_data=(x_valid,y_valid),
#          callbacks=[checkpoint],
#          verbose=2, shuffle=True)
filepath="weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]


classifier.fit_generator(training_set,
                         steps_per_epoch = 8000,
                         nb_epoch = 25,
                         validation_data = test_set,
                         validation_steps = 2000,
                         callbacks=callbacks_list                         
                         )

#num_of_test_samples = 800
#Confution Matrix and Classification Report
#Y_pred = classifier.predict_generator(test_set, num_of_test_samples // 17)
#y_pred = np.argmax(Y_pred, axis=1)
#print('Confusion Matrix')
#print(confusion_matrix(test_set.classes, y_pred))
#print('Classification Report')
#target_names = ['Cats', 'Dogs', 'Horse']
#print(classification_report(test_set.classes, y_pred, target_names=target_names))






filename = 'dogandcat3.sav'
pickle.dump(classifier, open(filename, 'wb'))

#from IPython.display import display
#from PIL import Image

#classifier.fit_generator(
 #                        training_set,
  #                       step_per_epoch = 8000,
   #                      epoch=10,
    #                     validation_data = test_set,
     #                    validation_steps= 800)



