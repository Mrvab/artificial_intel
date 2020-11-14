#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 18:53:51 2020

@author: sadrobin
"""

import numpy as np
from keras.preprocessing.image import load_img
import matplotlib.pyplot as plt
from keras.preprocessing import image
import pickle
#from keras.preprocessing.image import ImageDataGenerator
#need a model to load weight
#
loaded_model.load_weights('checkpoints/weights-improvement-17-0.80.hdf5')
filename='dogandcat.sav'
loaded_model = pickle.load(open(filename, 'rb'))
test_image = load_img('dataset/check/cat/1.jpg',target_size=(64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)

#test_datagen = ImageDataGenerator(rescale = 1./255)

#test_image =test_datagen.flow_from_directory('dataset/check',
 #                                           target_size = (64, 64),
  #                                          batch_size = 16,
   #                                         class_mode = 'binary')
loaded_model.summary()

#print(test_image[0][0][1])
#img = test_image[0][0][2].reshape((-1, 64, 64, 3))
plt.imshow(test_image)
plt.title("none")
plt.show()



result = loaded_model.predict(test_image)


if result[0][0] >= 0.5:
    print((result[0][0]))
    prediction = 'dog'
else:
    print((1-result[0][0]))
    prediction = 'cat'
print(prediction)