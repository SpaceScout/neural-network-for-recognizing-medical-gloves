#!/usr/bin/env python
# coding: utf-8

# In[83]:


print("a")


# In[84]:


import sklearn
import keras
from keras.models import Sequential
from keras.layers import Dense,Activation,Flatten,Dropout
from keras.layers import Conv2D,MaxPooling2D
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras.models import load_model
from sklearn.model_selection import train_test_split
import pandas as pd


import cv2
import os
import numpy as np
from matplotlib import pyplot as plt


# In[85]:


#загрузка датасетa

data_path='gloves dataset\dataset'
categories=os.listdir(data_path)
labels=[i for i in range(len(categories))]

label_dict=dict(zip(categories,labels))

print(label_dict)
print(categories)
print(labels)


# In[86]:


img_size=100
data=[]
target=[]


for category in categories:
    folder_path=os.path.join(data_path,category)
    img_names=os.listdir(folder_path)
        
    for img_name in img_names:
        img_path=os.path.join(folder_path,img_name)
        img=cv2.imread(img_path)

        try:
            gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)           
            #конвертируем изображение в серый 
            resized=cv2.resize(gray,(img_size,img_size))
            #ресайзим изображение в формат 100x100 
            data.append(resized)
            target.append(label_dict[category])
            #добавление изображения и метки(по категориям) и в список (набор данных)

        except Exception as e:
            print('Exception:',e)
            #если возникнет какая-либо ошибка, она будет напечатана здесь
            
print(img_size)


# In[87]:


data=np.array(data)/255.0
data=np.reshape(data,(data.shape[0],img_size,img_size,1))
target=np.array(target)

new_target=np_utils.to_categorical(target)


# In[88]:


np.save('data',data)
np.save('target',new_target)


# In[92]:


#архитектура сети

model=Sequential()

model.add(Conv2D(200,(3,3),input_shape=(100, 100,1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
#The first CNN layer followed by Relu and MaxPooling layers

model.add(Conv2D(100,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
#The second convolution layer followed by Relu and MaxPooling layers

model.add(Flatten())
model.add(Dropout(0.5))
#Flatten layer to stack the output convolutions from second convolution layer
model.add(Dense(50,activation='relu'))
#Dense layer of 50 neurons
model.add(Dense(2,activation='softmax'))
#The Final layer with two outputs for two categories

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

#model.summary()


# In[93]:


train_data,test_data,train_target,test_target=train_test_split(data,target,test_size=0.1)


# In[94]:


#обучение
checkpoint = ModelCheckpoint('model-{epoch:03d}.model',monitor='val_loss',verbose=0,save_best_only=True,mode='auto')
history=model.fit(train_data,train_target,epochs=20,callbacks=[checkpoint],validation_split=0.1)


# In[66]:


#график обучения
plt.plot(history.history['loss'],'r',label='training loss')
plt.plot(history.history['val_loss'],label='validation loss')
plt.xlabel('# epochs')
plt.ylabel('loss')
plt.legend()
plt.show()


# In[61]:


#настройки камеры для распознавая
model = load_model('model-015.model')

gloves_clsfr=cv2.CascadeClassifier(r'E:\anaconda\Lib\site-packages\cv2\data\haarcascade_palm.xml')

size = 4
camera=cv2.VideoCapture(0)

labels_dict={0:'No_Glove',1:'Glove'}
color_dict={0:(0,0,255),1:(0,255,0)}


# In[62]:


while True:
    ret,img=camera.read()
    gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gloves=gloves_clsfr.detectMultiScale(gray,scaleFactor=1.3,minNeighbors=3) 

    # Draw rectangles around each hand
    for (x,y,w,h) in gloves:
        #Rectangle around the hands
        hand_img = gray[y:y+w,x:x+w]
        resized=cv2.resize(hand_img,(100,100,1))
        normalized=resized/255.0
        reshaped=np.reshape(normalized,(1,100,100,1))
        reshaped = np.vstack([reshaped])
        result=model.predict(reshaped)
        #print(result)
        
        label=np.argmax(result,axis=1)[0]
      
        cv2.rectangle(img,(x,y),(x+w,y+h),color_dict[label],2)
        cv2.rectangle(img,(x,y-40),(x+w,y),color_dict[label],-1)
        cv2.putText(img, labels_dict[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        
    # Displaying the image
    cv2.imshow('LIVE',img)
    key = cv2.waitKey(30)
    # Exit via the escape key 
    if key == 27: 
        break
        
camera.release()
cv2.destroyAllWindows()


# In[ ]:




