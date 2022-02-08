# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 12:54:44 2022

@author: Jalpesh Dadania

This code uses 256x256 images/masks.
"""

from os import listdir
from keras.preprocessing.image import load_img
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from unnet_model import roadextract   #Use normal unet model
from sklearn.model_selection import train_test_split


# dataset path
X = "/content/drive/MyDrive/Colab Notebooks/road_segmentation_ideal/training/input/"
Y = "/content/drive/MyDrive/Colab Notebooks/road_segmentation_ideal/training/output/"

#list for image and mask
img = [] 
masks = [] 

#load images-Iterate through all images
def load_images(img = img,masks=masks, imgPath = None, maskPath = None, shape = 256):
  for filename in listdir(imgPath):
    pixels_X=load_img(imgPath+filename)
    pixels_X=cv2.resize(np.float32(pixels_X), (shape, shape))
    img.append(pixels_X)
    pixels_Y=load_img(maskPath+filename)
    pixels_Y=cv2.resize(np.float32(pixels_Y), (shape, shape))
    masks.append(pixels_Y[:,:,0])
  return img,masks

#normlize images
img = np.array(img) / 255.
masks = np.array(masks) 


#split the data 
X_train, X_test, y_train, y_test = train_test_split(img,masks, test_size = 0.20, random_state = 0)

#view the images
plt.subplot(1,2,1)
plt.imshow(X_train[1])
plt.show()
plt.subplot(1,2,2)
plt.imshow(y_train[1])
plt.show()

# call model
def get_model():
    inputs = tf.keras.layers.Input((256, 256, 3))
    model =roadextract(inputs, droupouts= 0.07)
    model.compile(optimizer = 'Adam', loss = 'binary_crossentropy', metrics = ['accuracy'] )
    return model


model = get_model()


#check point
earlystoping = tf.keras.callbacks.EarlyStopping(patience=2,monitor='val_loss')



#train the model

roadextract_training = model.fit(np.array(X_train),np.array(y_train),batch_size = 16, validation_data=(np.array(X_test), np.array(y_test)),  epochs = 10, verbose = 1,callbacks=earlystoping)

#save the model
model.save('mitochondria_test.hdf5')

# evaluate model
_, acc = model.evaluate(X_test, y_test)
print("Accuracy = ", (acc * 100.0), "%")

#plot the training and validation accuracy and loss at each epoch
loss = roadextract_training.history['loss']
val_loss = roadextract_training.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc = roadextract_training.history['accuracy']
val_acc = roadextract_training.history['val_accuracy']


plt.plot(epochs, acc, 'y', label='Training Accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
plt.title('Training and validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

#IOU
y_pred=roadextract.predict(X_test)
y_pred_thresholded = y_pred > 0.5

intersection = np.logical_and(y_test.flatten(),y_pred_thresholded.flatten())
union = np.logical_or(y_test.flatten(),y_pred_thresholded.flatten())
iou_score = np.sum(intersection) / np.sum(union)
print("IoU socre is: ", iou_score)

#extract the road on validation data
def predict (val_img,val_masks,model, shape = 256):
    ## getting and proccessing val data
    val_img = val_img[0:12]
    val_masks = val_masks[0:12]
    #mask = mask[0:16]
    
    imgProc = val_img [0:12]
    imgProc = np.array(val_img)
    
    predictions = model.predict(imgProc)
  

    return predictions, imgProc, val_masks


def Plotter(img, predMask, groundTruth):
    plt.figure(figsize=(9,9))
    
    plt.subplot(1,3,1)
    plt.imshow(img)
    plt.title('Satellite image')
    
    plt.subplot(1,3,2)
    plt.imshow(predMask)
    plt.title('Predicted Routes')
    
    plt.subplot(1,3,3)
    plt.imshow(groundTruth)
    plt.title('Actual Routes')
    

#Validation dataset
X_val = "/content/drive/MyDrive/Colab Notebooks/road_segmentation_ideal/testing/input/"
Y_val = "/content/drive/MyDrive/Colab Notebooks/road_segmentation_ideal/testing/output/"    
    

#list
val_img = [] 
val_masks = []

#preprocess validation dataset
def load_images_val(val_img = None,val_masks=None, imgPath = None, maskPath = None, shape = 256):
  for filename in listdir(imgPath):
    pixels_X=load_img(imgPath+filename)
    pixels_X=cv2.resize(np.float32(pixels_X), (shape, shape))
    val_img.append(pixels_X)
    pixels_Y=load_img(maskPath+filename)
    pixels_Y=cv2.resize(np.float32(pixels_Y), (shape, shape))
    val_masks.append(pixels_Y[:,:,0])
  return val_img,val_masks

val_img,val_masks = load_images_val(val_img = val_img,val_masks=val_masks,imgPath=X_val,maskPath = Y_val,shape = 256)

#normalizing validation dataset
val_img = np.array(val_img) / 255.
val_masks = np.array(val_masks) 

#check validation dataset
## displaying data loaded by our function
plt.subplot(1,2,1)
plt.imshow(val_img[1])
plt.show()
plt.subplot(1,2,2)
plt.imshow(val_masks[1])
plt.show()


#predict
predicted_Val, actuals, masks = predict(val_img, val_masks,roadextract)
#plot the results
Plotter(actuals[1], predicted_Val[1][:,:,0], masks[1])
Plotter(actuals[2], predicted_Val[2][:,:,0], masks[2])


