#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf
import time
from tensorflow.keras import datasets, layers, models

(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()  #mnist데이터로드

train_images = train_images.reshape((60000, 28, 28, 1)) #학습데이터
test_images = test_images.reshape((10000, 28, 28, 1)) #테스트데이터

# 픽셀 값을 0~1 사이로 정규화합니다.
train_images, test_images = train_images / 255.0, test_images / 255.0


model = models.Sequential()  #Keras의 Sequential API를 사용하여 CNN 모델을 정의
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))


model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


start = time.time()  #함수 시작 전 시간 측정
model.fit(train_images, train_labels, epochs=5)
print(time.time() - start)  #훈련에 걸린 시간 출력

