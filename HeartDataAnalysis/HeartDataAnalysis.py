# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 21:02:51 2021

@author: Alec
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from tensorflow import keras
from tensorflow.keras import layers

heart_data = pd.read_csv('heart.csv')
o2Saturation_data = pd.read_csv('o2Saturation.csv')

#sns.pairplot(heart_data)
#sns.heatmap(heart_data)

x= heart_data.drop(['output'], axis= 1)
y= heart_data[['output']]

normalized_X = preprocessing.normalize(x)
standardized_X = preprocessing.scale(x)
x_train, x_test, y_train, y_test = train_test_split(standardized_X, y, test_size=0.2, random_state=0)
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(13, input_shape=(13,)))
model.add(tf.keras.layers.Dense(6, activation='relu'))
model.add(tf.keras.layers.Dense(13, activation='relu'))
model.add(tf.keras.layers.Dense(6, activation='relu'))
model.add(tf.keras.layers.Dense(13, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=5, epochs= 10, steps_per_epoch=30, validation_data=(x_test, y_test))

test1 = x_test[[1]]
test2 = x_test[[30]]
test_prediction = model.predict(test1)
test_prediction1 = model.predict(test2)