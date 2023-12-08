#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('pip install -r requirements.txt')


# In[2]:


#!unzip Intel_Image_Classification_small.zip


# In[3]:


train_path = './Intel_Image_Classification_small/seg_train/seg_train/'
val_path = './Intel_Image_Classification_small/seg_test/seg_test/'
test_path = './Intel_Image_Classification_small/seg_pred/seg_pred/'


# In[4]:


import os
import pandas as pd


# In[5]:


DIM_SIZE = 299
class_name = os.listdir(train_path)
LAYER_NUM = len(class_name)


# In[6]:


import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf


# In[7]:


from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img


# In[7]:


#!unzip Intel_Image_Classification.zip


# In[8]:


from tensorflow.keras.preprocessing.image import load_img

path = './Intel_Image_Classification_small/seg_train/seg_train/buildings/'
name = '9807.jpg'
fullname = f'{path}/{name}'
load_img(fullname)


# In[9]:


img = load_img(fullname, target_size=(299, 299))


# In[10]:


img


# In[11]:


x = np.array(img)
x.shape


# ## Model

# Train and validade the Xception model:
# 
#     
# Using imagenet dataset

# In[12]:


from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.applications.xception import decode_predictions

from tensorflow.keras.preprocessing.image import ImageDataGenerator


# In[13]:


train_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_ds = train_gen.flow_from_directory(
    train_path,
    target_size=(DIM_SIZE, DIM_SIZE),
    batch_size=32
)


# In[14]:


val_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

val_ds = val_gen.flow_from_directory(
    val_path,
    target_size=(DIM_SIZE, DIM_SIZE),
    batch_size=32,
    shuffle=False
)


# # Xception

# In[15]:


from tensorflow.keras.applications.xception import Xception


# In[16]:


xception_model = Xception(
    weights='imagenet',
    include_top=False,
    input_shape=(DIM_SIZE, DIM_SIZE, 3)
)


# ### Training/Validating

# In[17]:


import numpy as np
import scipy


# In[18]:


EPOCHS = 5


# In[19]:


model_dict = {'xception_model': xception_model}


# In[20]:


checkpoint = keras.callbacks.ModelCheckpoint(
    'xception_{epoch:02d}_{val_accuracy:.3f}.h5',
    save_best_only=True,
    monitor='val_accuracy',
    mode='max'
)


# In[21]:


scores = {}

def train_val_model(model_dict):

    for model_name in model_dict:
        
        base_model = model_dict[model_name]
        
        base_model.trainable = False

        inputs = keras.Input(shape=(DIM_SIZE, DIM_SIZE, 3))

        base = base_model(inputs, training=False)

        vectors = keras.layers.GlobalAveragePooling2D()(base)

        outputs = keras.layers.Dense(LAYER_NUM)(vectors)

        model = keras.Model(inputs, outputs)

        learning_rate = 0.01
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

        loss = keras.losses.CategoricalCrossentropy(from_logits=True)

        model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

        history = model.fit(train_ds, epochs=EPOCHS, validation_data=val_ds, callbacks=[checkpoint])
        
        scores = history.history
        
    return scores


# In[23]:


#scores = train_val_model(model_dict)


# In[23]:


scores.items()


# In[22]:


for key, value in scores.items():
    if (key == 'val_accuracy'):
        val_accuracy = value
val_accuracy        


# In[ ]:


scores.items()


# In[26]:


for key, value in scores.items():
    if (key == 'val_accuracy'):
        plt.plot(val_accuracy, label=('accuracy'))
    #plt.plot(hist['val_accuracy'], label=(f'lr={lr}'))

plt.xticks(np.arange(EPOCHS))
plt.legend()


# 
# ## Parameter Tuning
# 
#     Learning Rate
#     Inner Size
#     Augmentation

# In[22]:


checkpoint1 = keras.callbacks.ModelCheckpoint(
    'xception2_{epoch:02d}_{val_accuracy:.4f}.h5',
    save_best_only=True,
    monitor='val_accuracy',
    mode='max'
)


# In[23]:


def train_xception(base_model, lr=0.001, inner_size=1000):
  
    base_model.trainable = False

    inputs = keras.Input(shape=(DIM_SIZE, DIM_SIZE, 3))

    base = base_model(inputs, training=False)

    vectors = keras.layers.GlobalAveragePooling2D()(base)

    inner = keras.layers.Dense(inner_size, activation='relu')(vectors)

    outputs = keras.layers.Dense(LAYER_NUM)(inner)

    model = keras.Model(inputs, outputs)

    optimizer = keras.optimizers.Adam(learning_rate=lr)

    loss = keras.losses.CategoricalCrossentropy(from_logits=True)

    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    history = model.fit(train_ds, epochs=EPOCHS, validation_data=val_ds, callbacks=[checkpoint1])
    
    return history.history


# # Learning Rate

# In[ ]:


# scores = {}
# for lr in [0.0001, 0.001, 0.01, 0.1]:
#     scores[lr] = train_xception(xception_model, lr)


# In[24]:


# scores = {}
# for lr in [0.001, 0.01, 0.1]:
#     scores[lr] = train_xception(xception_model, lr)


# In[25]:


for lr, hist in scores.items():
    plt.plot(hist['val_accuracy'], label=(lr))

plt.xticks(np.arange(EPOCHS))
plt.legend()


# ## Best learning rate for model : 0.001

# # Inner Size

# In[ ]:


# scores = {}

# for size in [10, 100, 1000]:
#     scores[size] = train_xception(xception_model, inner_size=size)


# In[ ]:


# scores = {}

# for size in [100, 1000]:
#     scores[size] = train_xception(xception_model, inner_size=size)


# In[25]:


scores = {}

for size in [1000]:
    scores[size] = train_xception(xception_model, inner_size=size)


# ## Best inner_size for model : 1000

# # Augmentation

# In[24]:


# Create image generator for train data and also augment the images
train_gen = ImageDataGenerator(preprocessing_function=preprocess_input,
                               rotation_range=30,
                               shear_range=10,
                               zoom_range=0.1)

train_ds = train_gen.flow_from_directory(train_path,
                                         target_size=(DIM_SIZE,DIM_SIZE),
                                         batch_size=32)


# In[26]:


scores = []
scores = train_xception(xception_model)


# In[40]:


for key, value in scores.items():    
    if (key == 'val_accuracy'): 
        print (value)
        
# 'val_accuracy': [0.8951559662818909,
#   0.9017916321754456,
#   0.9137359261512756,
#   0.9031187891960144,
#   0.9090909361839294]        


# In[41]:


scores
for key, value in scores.items():    
    if (key == 'val_accuracy'):
        plt.plot(value, label=('accuracy'))

plt.xticks(np.arange(EPOCHS))
plt.legend()


# ## Training again for best model and saving (last time)

# In[25]:


EPOCHS = 10


# In[26]:


def train_xception2(base_model, lr=0.001, inner_size=1000):
  
    base_model.trainable = False

    inputs = keras.Input(shape=(DIM_SIZE, DIM_SIZE, 3))

    base = base_model(inputs, training=False)

    vectors = keras.layers.GlobalAveragePooling2D()(base)

    inner = keras.layers.Dense(inner_size, activation='relu')(vectors)

    outputs = keras.layers.Dense(LAYER_NUM)(inner)

    model = keras.Model(inputs, outputs)

    optimizer = keras.optimizers.Adam(learning_rate=lr)

    loss = keras.losses.CategoricalCrossentropy(from_logits=True)

    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    return model


# In[29]:


model = train_xception(xception_model)

#history = model.fit(train_ds, epochs=EPOCHS, validation_data=val_ds, callbacks=[checkpoint3]) #this was a mistake


# ## Saved the best model 'xception2_08_0.9283.h5'
