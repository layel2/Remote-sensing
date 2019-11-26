from osgeo import gdal, gdalconst
from osgeo.gdalconst import *
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix,cohen_kappa_score
from tensorflow import keras
from PIL import Image

img_path = './FinalQ4Image.jpg'

ds = gdal.Open(img_path, GA_ReadOnly)
n_band = ds.RasterCount
img_width = 2448
img_height = 2448
img = np.zeros((img_width,img_height,n_band))
for i in range(n_band):
    band = ds.GetRasterBand(i+1)
    data = band.ReadAsArray()
    img[:,:,i] = data
data = None
band = None
img = (img)/255

pointWater = pd.read_csv('./water_samples.csv').values
pointUrban = pd.read_csv('./urban_samples.csv').values
pointAgri = pd.read_csv('./agriculture_samples.csv').values
pointRange = pd.read_csv('./Range_samples.csv').values
pointBarren = pd.read_csv('./barren_samples.csv').values

def imgPointToData(img,point):
    dataOut = []
    for i in range(len(point)):
        dataOut.append(img[point[i,0],point[i,1]])
    return np.array(dataOut)

dataWater = imgPointToData(img,pointWater)
dataUrban = imgPointToData(img,pointUrban)
dataAgri = imgPointToData(img,pointAgri)
dataRange = imgPointToData(img,pointRange)
dataBarren = imgPointToData(img,pointBarren)

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

#print(X_train.shape)
#print(y_train.shape)
#print(X_test.shape)
#print(y_test.shape)

idx1 = np.random.permutation(len(X_train))
X_train = X_train[idx1]
y_train = y_train[idx1]

idx2 = np.random.permutation(len(X_test))
X_test = X_test[idx2]
y_test = y_test[idx2]


X_train1 = np.concatenate((dataWater[:8000],dataUrban[:8000],dataAgri[:8000],dataRange[:8000],dataBarren[:8000]))
y_train1 = np.ones((len(X_train)))
y_train1[:8000] *= 0
y_train1[8000:40000] *= 1


X_test1  = np.concatenate((dataWater[8000:],dataUrban[8000:],dataAgri[8000:],dataRange[8000:],dataBarren[8000:]))
y_test1 = np.ones((len(X_test1)))
y_test1[:2000] *= 0
y_test1[2000:10000] *= 1

idx1 = np.random.permutation(len(X_train1))
X_train1 = X_train1[idx1]
y_train1 = y_train1[idx1]

idx2 = np.random.permutation(len(X_test1))
X_test1 = X_test1[idx2]
y_test1 = y_test1[idx2]


X_train2 = np.concatenate((dataUrban[:8000],dataAgri[:8000],dataRange[:8000],dataBarren[:8000]))
y_train2 = np.ones((len(X_train2)))
y_train2[:8000] *= 0
y_train2[8000:16000] *= 1
y_train2[16000:24000] *= 2
y_train2[24000:36000] *= 3


X_test2 = np.concatenate((dataUrban[8000:],dataAgri[8000:],dataRange[8000:],dataBarren[8000:]))
y_test2 = np.ones((len(X_test2)))
y_test2[:2000] *= 0
y_test2[2000:4000] *= 1
y_test2[4000:6000] *= 2
y_test2[6000:8000] *= 3


idx1 = np.random.permutation(len(X_train2))
X_train2 = X_train2[idx1]
y_train2 = y_train2[idx1]

idx2 = np.random.permutation(len(X_test2))
X_test2 = X_test2[idx2]
y_test2 = y_test2[idx2]


y_train_encode = keras.utils.to_categorical(y_train)
y_test_encode = keras.utils.to_categorical(y_test)
y_train_encode2 = keras.utils.to_categorical(y_train2)
y_test_encode2 = keras.utils.to_categorical(y_test2)


def nnmodel(input_shape):
    X_input = keras.layers.Input((input_shape))
    X = keras.layers.Dense(128,activation='relu')(X_input)
    X = keras.layers.Dense(32,activation='relu')(X)
    X = keras.layers.Dense(1,activation='sigmoid')(X)
    model = keras.models.Model(inputs=X_input, outputs=X, name='model')
    return model


mymodel = nnmodel(X_train1[0].shape)
mymodel.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])
mymodel.summary()

print("Train 1st model")
mymodel.fit(X_train1,y_train1,batch_size=32,epochs=10,validation_data=(X_test1,y_test1))


his = mymodel.history.history
plt.plot(his['acc'])
plt.plot(his['val_acc'])

pred1 = mymodel.predict(X_test1)
pred1 = (pred1>0.5)*1

#print(classification_report(y_test1,pred1))

def nnmodel2(input_shape):
    X_input = keras.layers.Input((input_shape))
    X = keras.layers.Dense(1024,activation='relu')(X_input)
    X = keras.layers.Dense(256,activation='relu')(X)
    X = keras.layers.Dense(4,activation='softmax')(X)
    model = keras.models.Model(inputs=X_input, outputs=X, name='model')
    return model


mymodel2 = nnmodel2(X_train2[0].shape)
opt = keras.optimizers.Adam(lr=0.0005)
mymodel2.compile(optimizer=opt, loss="categorical_crossentropy", metrics=['accuracy'])
mymodel2.summary()

print("Train 2nd model")
mymodel2.fit(X_train2,y_train_encode2,batch_size=32,epochs=50,validation_data=(X_test2,y_test_encode2))

his2 = mymodel2.history.history
plt.plot(his2['acc'])
plt.plot(his2['val_acc'])


pred2 = np.argmax(mymodel2.predict(X_test2),axis=1)

#print(classification_report(y_test2,pred2))

def commodel(data):
    l1 = np.round(mymodel.predict(data))
    l2 = np.argmax(mymodel2.predict(data),axis=1)
    outdata = []
    for i in range(len(data)):
        if(l1[i]==0):
            outdata.append(0)
        else:
            outdata.append(l2[i]+1)
    return np.array(outdata)

pred = commodel(X_test)

#print(classification_report(y_test,pred))

img = img.reshape(((2448* 2448, 3)))
#img = img*255

predimg = commodel(img)
predimg_dec = predimg
colormap = np.zeros_like(img).astype(int)

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


colormap = colormap.reshape((2448,2448,3))


plt.figure(figsize=(10,10))
plt.imshow(colormap)


im = Image.fromarray(colormap.astype(np.uint8))
im.save('./final4-lcm.png')
print('saved "./final4-lcm.png" ')

confmat = confusion_matrix(y_test,pred)
print('----- Confusion matrix -----')
print(confmat)


user_acc = [confmat[0,0],confmat[1,1],confmat[2,2],confmat[3,3],confmat[4,4]]/np.sum(confmat,axis=1)*100
print(f'User accuracy : {user_acc} %')


prod_acc = [confmat[0,0],confmat[1,1],confmat[2,2],confmat[3,3],confmat[4,4]]/np.sum(confmat,axis=0)*100
print(f'Producer accuracy : {prod_acc} %')


overall_acc = np.sum([confmat[0,0],confmat[1,1],confmat[2,2],confmat[3,3],confmat[4,4]])/100
print(f'Overall accuracy : {overall_acc} %')

(overall_acc-1/5)/(1-1/5)


kappa = cohen_kappa_score(y_test,pred)
print(f'Kappa Score : {kappa}')


#print(classification_report(y_test,pred))