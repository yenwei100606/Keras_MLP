# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 17:52:02 2019

@author: Yen_Wei
"""


import cv2,glob
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model

files = glob.glob("imagedata\*.jpg")
test_feature=[]
test_label=[]

def show_images_label_predictions(images,labels,predictions,start_id,num=10):
    plt.gcf().set_size_inches(12,14)
    if num>25:
        num=25
    for i in range(0,num):
        ax = plt.subplot(5, 5,1+i)
        ax.imshow(images[start_id],cmap='binary')  #顯示黑白
        
        #有預測結果才在標題顯現
        if(len(predictions) > 0):
            title = 'ai=' + str(predictions[i])
            title += ('(o)' if predictions[i]==labels[i] else '(x)')
            title += '\nlabel=' + str(labels[i])
        else:
            title = 'label=' + str(labels[i])
        
        ax.set_title(title,fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
        start_id+=1
        
    plt.show()

for file in files:
    img = cv2.imread(file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _,img = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY_INV)
    test_feature.append(img)
    label = file[10:11]
    test_label.append(int(label))
    
test_feature = np.array(test_feature)
test_label = np.array(test_label)

test_feature_vector = test_feature.reshape(len(test_feature),784).astype('float32')

test_feature_normalize = test_feature_vector/255

print("載入模型Minst_MLP_model.h5")
model = load_model('Minst_MLP_model.h5')

prediction = model.predict_classes(test_feature_normalize)

show_images_label_predictions(test_feature, test_label, prediction, 0)

