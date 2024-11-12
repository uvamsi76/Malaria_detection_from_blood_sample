import os
import tensorflow as tf
import cv2
import random
from tensorflow.keras.layers import InputLayer,Conv2D,MaxPool2D,Dense,Flatten,BatchNormalization
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam

IM_size=224
lenet_model=tf.keras.Sequential([
    InputLayer(shape=(IM_size,IM_size,3)),

    Conv2D(filters=6,kernel_size=3,strides=1,padding='valid',activation='relu'),
    BatchNormalization(),
    MaxPool2D(pool_size=2,strides=2),

    Conv2D(filters=16,kernel_size=3,strides=1,padding='valid',activation='relu'),
    BatchNormalization(),
    MaxPool2D(pool_size=2,strides=2),

    Flatten(),

    Dense(100,activation='relu'),
    BatchNormalization(),
    Dense(10,activation='relu'),
    BatchNormalization(),
    Dense(1,activation='sigmoid')
])

def train(train_ds,val_ds,model,epochs,model_version="1.0.0",lr=0.01):
    model.compile(optimizer=Adam(learning_rate=lr),
              loss=BinaryCrossentropy(),
              metrics=["accuracy"]
              )
    history=model.fit(train_ds,epochs=epochs,verbose=1)
    model.save(f'malaria_detection/models/Lenet_model{model_version}.h5')
    return model