# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense

def show_image(image):
    fig = plt.gcf()
    fig.set_size_inches(2,4)
    plt.imshow(image,cmap='binary')
    plt.show()

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
(train_feature,train_label),(test_feature,test_label) = mnist.load_data()

#show_image(train_feature[10])

train_feature_vector = train_feature.reshape(len(train_feature),28*28)
test_feature_vector = test_feature.reshape(len(test_feature),28*28)

#特徵值標準化
train_feature_normalize = train_feature_vector/255
test_feature_normalize = test_feature_vector/255

#label 轉換成 onehot encoding
train_label_onehot = np_utils.to_categorical(train_label)
test_label_onehot = np_utils.to_categorical(test_label)

#建立模型 
model = Sequential()
model.add(Dense(input_dim=28*28,units=256,activation='relu'))
model.add(Dense(units=256,activation='relu'))
model.add(Dense(units=256,activation='relu'))
model.add(Dense(units=10,activation='softmax'))


model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])

#model.compile(optimizer='adam', loss='mse',metrics=['accuracy'])


train_history = model.fit(x=train_feature_normalize,y=train_label_onehot,validation_split=0.2,epochs=10,batch_size=100)

scores = model.evaluate(x=train_feature_normalize,y=train_label_onehot)
print("Training Acc:",scores[1])

scores = model.evaluate(x=test_feature_normalize,y=test_label_onehot)
print("Testing Acc:",scores[1])

#將模型儲存至HDF5
model.save('Minst_MLP_model.h5')
print("模型儲存完畢")
#將參數儲存不包含模型
model.save_weights("Minst_MLP_model2.weight")
print("參數儲存完畢")

prediction = model.predict_classes(test_feature_normalize)

show_images_label_predictions(test_feature, test_label, prediction, 0)