#!/usr/bin/env python
# coding: utf-8

# In[1]:


from osgeo import gdal, gdalconst
from osgeo.gdalconst import *
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd


# In[2]:


img_path = './FinalQ4Image.jpg'


# In[3]:


ds = gdal.Open(img_path, GA_ReadOnly)


# In[4]:


n_band = ds.RasterCount


# In[5]:


img_width = 2448
img_height = 2448


# In[6]:


img = np.zeros((img_width,img_height,n_band))
for i in range(n_band):
    band = ds.GetRasterBand(i+1)
    data = band.ReadAsArray()
    img[:,:,i] = data
data = None
band = None


# In[7]:


img.shape


# In[12]:


pointWater = pd.read_csv('./water_samples.csv').values
pointUrban = pd.read_csv('./urban_samples.csv').values
pointAgri = pd.read_csv('./agriculture_samples.csv').values
pointRange = pd.read_csv('./Range_samples.csv').values
pointBarren = pd.read_csv('./barren_samples.csv').values


# In[25]:


def imgPointToData(img,point):
    dataOut = []
    for i in range(len(point)):
        dataOut.append(img[point[i,0],point[i,1]])
    return np.array(dataOut)


# In[28]:


dataWater = imgPointToData(img,pointWater)
dataUrban = imgPointToData(img,pointUrban)
dataAgri = imgPointToData(img,pointAgri)
dataRange = imgPointToData(img,pointRange)
dataBarren = imgPointToData(img,pointBarren)


# In[106]:


X_train = np.concatenate((dataWater[:8000],dataUrban[:8000],dataAgri[:8000],dataRange[:8000],dataBarren[:8000]))
y_train = np.ones((len(X_train)))
y_train[:8000] *= 0
y_train[8000:16000] *= 1
y_train[16000:24000] *= 2
y_train[24000:36000] *= 3
y_train[36000:40000] *= 4

X_test = np.concatenate((dataWater[8000:],dataUrban[8000:],dataAgri[8000:],dataRange[8000:],dataBarren[8000:]))
y_test = np.ones((len(X_test)))
y_test[:2000] *= 0
y_test[2000:4000] *= 1
y_test[4000:6000] *= 2
y_test[6000:8000] *= 3
y_test[8000:10000] *= 4


# In[107]:


print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# In[129]:


idx1 = np.random.permutation(len(X_train))
X_train = X_train[idx1]
y_train = y_train[idx1]

idx2 = np.random.permutation(len(X_test))
X_test = X_test[idx2]
y_test = y_test[idx2]

X_train = X_train/255
X_test = X_test/255


# In[114]:


y_train[2002:]


# In[118]:


import sklearn
import xgboost as xgb


# In[120]:


from sklearn.metrics import classification_report


# In[130]:


xgbc = xgb.XGBClassifier(max_depth=21,n_estimators=300)
xgbc.fit(X_train,y_train)


# In[131]:


print(classification_report(y_test,xgbc.predict(X_test)))


# In[144]:


from sklearn.ensemble import RandomForestClassifier


# In[148]:


rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train,y_train)


# In[149]:


print(classification_report(y_test,rfc.predict(X_test)))


# In[150]:


from sklearn.svm import SVC


# In[154]:


svmc = SVC(C=5)
svmc.fit(X_train,y_train)


# In[155]:


print(classification_report(y_test,svmc.predict(X_test)))


# In[127]:


from tensorflow import keras


# In[132]:


y_train_encode = keras.utils.to_categorical(y_train)
y_test_encode = keras.utils.to_categorical(y_test)


# In[156]:


def nnmodel(input_shape):
    X_input = keras.layers.Input((input_shape))
    X = keras.layers.Dense(1024,activation='relu')(X_input)
    X = keras.layers.Dense(128,activation='relu')(X)
    X = keras.layers.Dense(5,activation='softmax')(X)
    model = keras.models.Model(inputs=X_input, outputs=X, name='model')
    return model


# In[157]:


mymodel = nnmodel(X_train[0].shape)
mymodel.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=['accuracy'])
mymodel.summary()


# In[158]:


mymodel.fit(X_train,y_train_encode,batch_size=32,epochs=20,validation_data=(X_test,y_test_encode))


# In[138]:


X_train[0]


# In[163]:


pred = np.argmax(mymodel.predict(X_test),axis=1)


# In[165]:


print(classification_report(y_test,pred))


# In[168]:


mymodel.evaluate(X_test,y_test_encode)


# In[169]:


pred.shape


# In[210]:


img = img.reshape(((2448* 2448, 3)))
img = img/255


# In[211]:


predimg = rfc.predict(img)


# In[223]:


#predimg_dec = np.argmax(predimg,axis=1)
predimg_dec = predimg


# In[224]:


colormap = np.zeros_like(img).astype(int)


# In[225]:


for i in range(len(colormap)):
    if(predimg_dec[i] == 0):
        #water
        colormap[i] = [0,0,255]
    elif(predimg_dec[i]==1):
        #urban
        colormap[i] = [0,255,255]
    elif (predimg_dec[i] ==2):
        #argi
        colormap[i] = [255,255,0]
    elif (predimg_dec[i] == 3):
        #range
        colormap[i] = [255,0,255]
    elif (predimg_dec[i] == 4):
        #barren
        colormap[i] = [255,255,255]


# In[226]:


predimg[0]


# In[227]:


colormap = colormap.reshape((2448,2448,3))


# In[228]:


plt.imshow(colormap)


# In[212]:


plt.hist(predimg)


# In[232]:


from sklearn.metrics import confusion_matrix,cohen_kappa_score


# In[235]:


confmat = confusion_matrix(y_test,pred)
confmat


# In[249]:


user_acc = [confmat[0,0],confmat[1,1],confmat[2,2],confmat[3,3],confmat[4,4]]/np.sum(confmat,axis=1)*100
user_acc


# In[252]:


prod_acc = [confmat[0,0],confmat[1,1],confmat[2,2],confmat[3,3],confmat[4,4]]/np.sum(confmat,axis=0)*100
prod_acc


# In[261]:


overall_acc = np.sum([confmat[0,0],confmat[1,1],confmat[2,2],confmat[3,3],confmat[4,4]])/10000
overall_acc


# In[263]:


(overall_acc-1/5)/(1-1/5)


# In[233]:


cohen_kappa_score(y_test,pred)


# In[254]:


print(classification_report(y_test,pred))


# In[243]:


confmat[0]


# In[251]:


np.sum(confmat,axis=0)


# In[ ]:




