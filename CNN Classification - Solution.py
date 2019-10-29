#!/usr/bin/env python
# coding: utf-8

# Diseases classification in Apple leaves
# 


import numpy as np
import keras
from keras import models
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.optimizers import SGD
from keras import layers
from keras.layers import  Dense, Flatten, Conv2D, MaxPooling2D
from keras import Input
                                                                                                            



train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

 
train_generator = train_datagen.flow_from_directory(
        'plant_village/train/',
        target_size=(64, 64),
        batch_size=16,
        class_mode='categorical')


validation_generator = validation_datagen.flow_from_directory(
        'plant_village/val/',
        target_size=(64, 64),
        batch_size=16,
        class_mode='categorical',
        shuffle=False)

test_generator = test_datagen.flow_from_directory(
        'plant_village/test/',
        target_size=(64, 64),
        batch_size=1,
        class_mode='categorical',
        shuffle=False)




model = models.Sequential()

model.add(Conv2D(128, kernel_size=(3,3), activation = 'relu', input_shape=(64,64,3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64, kernel_size=(3,3), activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64, kernel_size=(3,3), activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(4, activation='softmax'))
model.summary()





sgd = SGD(lr=0.001,decay=1e-6, momentum=0.9, nesterov=True)

model.compile(sgd, loss='categorical_crossentropy', metrics=['acc'])

history = model.fit_generator(train_generator, 
      steps_per_epoch=train_generator.samples/train_generator.batch_size,
      epochs=10,
      validation_data=validation_generator,
      validation_steps=validation_generator.samples/validation_generator.batch_size,
      verbose=1)







model.save('cnn_classification.h5')





model = models.load_model('cnn_classification.h5')
print(model)







model.save_weights('cnn_classification.h5')







model.load_weights('cnn_classification.h5')




train_acc = history.history['acc']
val_acc = history.history['val_acc']
train_loss = history.history['loss']
val_loss = history.history['val_loss']
print(train_acc)
print(val_acc)
print(train_loss)
print(val_loss)





epochs = range(len(train_acc)) 
plt.plot(epochs, train_acc, 'b', label='Training Accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.figure()
plt.show()

plt.plot(epochs, train_loss, 'b', label='Training Loss')
plt.plot(epochs, val_loss, 'r', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()



fnames = test_generator.filenames
 

ground_truth = test_generator.classes
 

label2index = test_generator.class_indices
 

idx2label = dict((v,k) for k,v in label2index.items())
 

predictions = model.predict_generator(test_generator, steps=test_generator.samples/test_generator.batch_size,verbose=1)
predicted_classes = np.argmax(predictions,axis=1)
 
errors = np.where(predicted_classes != ground_truth)[0]
print("No of errors = {}/{}".format(len(errors),test_generator.samples))





new_model= models.Sequential()
model.load_weights('cnn_classification.h5', by_name=True)
new_model.add(Conv2D(128, kernel_size=(3,3), activation = 'relu', input_shape=(64,64,3)))
new_model.add(MaxPooling2D(pool_size=(2,2)))
new_model.add(Conv2D(64, kernel_size=(3,3), activation = 'relu'))
new_model.add(MaxPooling2D(pool_size=(2,2)))
new_model.add(Conv2D(64, kernel_size=(3,3), activation = 'relu'))
new_model.add(MaxPooling2D(pool_size=(2,2)))
new_model.add(Conv2D(32, kernel_size=(3,3), activation = 'relu'))
new_model.add(Conv2D(32, kernel_size=(3,3), activation = 'relu'))
new_model.add(layers.Flatten())
new_model.add(layers.Dense(32, activation='relu'))
new_model.add(layers.Dense(4, activation='softmax'))
new_model.summary()





for layer in new_model.layers[:6]:
    layer.trainable = False

for layer in new_model.layers:
    print(layer, layer.trainable)
new_model.summary()





sgd = SGD(lr=0.001,decay=1e-6, momentum=0.9, nesterov=True)

new_model.compile(sgd, loss='categorical_crossentropy', metrics=['acc'])
 
new_history = new_model.fit_generator(train_generator, 
      steps_per_epoch=train_generator.samples/train_generator.batch_size,
      epochs=2,
      validation_data=validation_generator,
      validation_steps=validation_generator.samples/validation_generator.batch_size,
      verbose=1)




train_acc = new_history.history['acc']
val_acc = new_history.history['val_acc']
train_loss = new_history.history['loss']
val_loss = new_history.history['val_loss']





epochs = range(len(train_acc)) 
plt.plot(epochs, train_acc, 'b', label='Training Accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.figure()
plt.show()

plt.plot(epochs, train_loss, 'b', label='Training Loss')
plt.plot(epochs, val_loss, 'r', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()








fnames = test_generator.filenames
 

ground_truth = test_generator.classes
 

label2index = test_generator.class_indices
 

idx2label = dict((v,k) for k,v in label2index.items())
 

predictions = model.predict_generator(test_generator, steps=test_generator.samples/test_generator.batch_size,verbose=1)
predicted_classes = np.argmax(predictions,axis=1)
 
errors = np.where(predicted_classes != ground_truth)[0]
print("No of errors = {}/{}".format(len(errors),test_generator.samples))





def new_image(file_path):
    new_datagen = ImageDataGenerator(rescale=1./255)
    new_generator = new_datagen.flow_from_directory(
        file_path,
        target_size=(64, 64),
        batch_size=1,
        class_mode='categorical',
        shuffle=False)
    fnames = new_generator.filenames
    ground_truth = new_generator.classes
    label2index = new_generator.class_indices
    idx2label = dict((v,k) for k,v in label2index.items())
    predictions1 = model.predict_generator(new_generator, steps=new_generator.samples/new_generator.batch_size,verbose=1)
    predicted_classes1 = np.argmax(predictions1,axis=1)
    print(predicted_classes1)
   
    
    











