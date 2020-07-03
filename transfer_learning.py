import time

import pandas as pd
import numpy as np
import os
import keras
import matplotlib.pyplot as plt
from keras.layers import Dense, GlobalAveragePooling2D
from keras.applications import MobileNet, Xception, VGG16, NASNetMobile
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.optimizers import Adam

mobileNet_model = MobileNet(weights='imagenet',include_top=False)  # imports the mobilenet model and discards the last 1000 neuron layer.
xception_model = Xception(include_top=False, weights="imagenet",)
vgg16_model = VGG16(include_top=False, weights="imagenet",)
NASNetMobile_model = NASNetMobile(include_top=False, weights="imagenet",)

base_model = mobileNet_model

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(
    x)  # we add dense layers so that the model can learn more complex functions and classify for better results.
x = Dense(1024, activation='relu')(x)  # dense layer 2
x = Dense(512, activation='relu')(x)  # dense layer 3
preds = Dense(102, activation='softmax')(x)  # final layer with softmax activation

# In[3]:


model = Model(inputs=base_model.input, outputs=preds)
# specify the inputs
# specify the outputs
# now a model has been created based on our architecture


# In[4]:


for layer in model.layers[:20]:
    layer.trainable = False
for layer in model.layers[20:]:
    layer.trainable = True

# In[5]:


train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input, validation_split=0.9)  # included in our dependencies

train_generator = train_datagen.flow_from_directory('data/organized_flowers_photos/train/',
                                                    # this is where you specify the path to the main data folder
                                                    target_size=(224, 224),
                                                    color_mode='rgb',
                                                    batch_size=32,
                                                    class_mode='categorical',
                                                    shuffle=True,
                                                    subset='training')

validation_generator = train_datagen.flow_from_directory('data/organized_flowers_photos/train/',
                                                    # this is where you specify the path to the main data folder
                                                    target_size=(224, 224),
                                                    color_mode='rgb',
                                                    batch_size=32,
                                                    class_mode='categorical',
                                                    shuffle=True,
                                                    subset='validation')

# In[33]:


model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
# Adam optimizer
# loss function will be categorical cross entropy
# evaluation metric will be accuracy

step_size_train = train_generator.n // train_generator.batch_size
step_size_validation = validation_generator.n // validation_generator.batch_size

start_time = time.time()
#model.fit_generator(generator=train_generator,
#                    steps_per_epoch=step_size_train,
#                    epochs=2)
model.fit_generator(
    train_generator,
    steps_per_epoch=step_size_train,
    validation_data=validation_generator,
    validation_steps=step_size_validation,
    epochs=1)
end_time = time.time()
print("total train time =", round(end_time - start_time, 2), 'seconds')

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)  # included in our dependencies
test_generator = test_datagen.flow_from_directory('data/organized_flowers_photos/test/',
                                                    # this is where you specify the path to the main data folder
                                                    target_size=(224, 224),
                                                    color_mode='rgb',
                                                    batch_size=32,
                                                    class_mode='categorical',
                                                    shuffle=True)

probabilities = model.predict_generator(test_generator)

from sklearn.metrics import accuracy_score

y_true = np.array(test_generator.labels)
y_pred = [np.argmax(x) for x in probabilities]

print("test accuracy =", accuracy_score(y_true, y_pred))
