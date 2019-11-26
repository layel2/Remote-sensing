import numpy as np
from matplotlib import pyplot as plt
import scipy.io
import pandas as pd
from sklearn.decomposition import PCA

paviaU = scipy.io.loadmat('./PaviaU.mat')['paviaU']
paviaU_gt = scipy.io.loadmat('./PaviaU_gt.mat')['paviaU_gt']

Nmax = paviaU.max()
paviaU = paviaU/Nmax

paviaU = paviaU.reshape(610*340, 103)

pca = PCA(3)
paviaU_PCA = pca.fit_transform(paviaU)

paviaU_PCA = paviaU_PCA.reshape((610,340,3))

paviaU.shape

for i in range(3):
    paviaU_PCA[:,:,i] = (paviaU_PCA[:,:,i]-paviaU_PCA[:,:,i].min())/(paviaU_PCA[:,:,i].max()-paviaU_PCA[:,:,i].min())
    paviaU_PCA[:,:,i] = paviaU_PCA[:,:,i]*255
paviaU_PCA = paviaU_PCA.astype(np.uint8)

plt.figure(figsize=(20,10))
plt.imshow(paviaU_PCA)

from PIL import Image
im = Image.fromarray(paviaU_PCA.astype(np.uint8))
im.save('./final5-3bandMin.png')

#plt.figure(figsize=(20,10))
#plt.imshow(paviaU_gt)

n_class,num_pre_class = np.unique(paviaU_gt.reshape((610*340)), return_counts=True)

pca2 = PCA(50)
paviaU_PCA2 = pca2.fit_transform(paviaU)

df = pd.DataFrame(paviaU_PCA2,paviaU_gt.reshape(610*340))
df = df.sample(frac=1)
df = df.sort_index()

ttc = np.floor(num_pre_class/10).astype(int)

X_train = np.concatenate((df.iloc[0:ttc[0]],
                          df.iloc[num_pre_class[:1].sum():num_pre_class[:1].sum()+ttc[1]],
                          df.iloc[num_pre_class[:2].sum():num_pre_class[:2].sum()+ttc[2]],
                         df.iloc[num_pre_class[:3].sum():num_pre_class[:3].sum()+ttc[3]],
                         df.iloc[num_pre_class[:4].sum():num_pre_class[:4].sum()+ttc[4]],
                         df.iloc[num_pre_class[:5].sum():num_pre_class[:5].sum()+ttc[5]],
                         df.iloc[num_pre_class[:6].sum():num_pre_class[:6].sum()+ttc[6]],
                         df.iloc[num_pre_class[:7].sum():num_pre_class[:7].sum()+ttc[7]],
                         df.iloc[num_pre_class[:8].sum():num_pre_class[:8].sum()+ttc[8]],
                         df.iloc[num_pre_class[:9].sum():num_pre_class[:9].sum()+ttc[9]]))

y_train = np.concatenate(( np.zeros(ttc[0]) ,
                        np.ones(ttc[1]),
                        np.ones(ttc[2])*2,
                        np.ones(ttc[3])*3,
                        np.ones(ttc[4])*4,
                        np.ones(ttc[5])*5,
                        np.ones(ttc[6])*6,
                        np.ones(ttc[7])*7,
                        np.ones(ttc[8])*8,
                        np.ones(ttc[9])*9,))

X_test = np.concatenate((df.iloc[ttc[0]:num_pre_class[:1].sum()],
                        df.iloc[num_pre_class[:1].sum()+ttc[1]:num_pre_class[:2].sum()],
                        df.iloc[num_pre_class[:2].sum()+ttc[2]:num_pre_class[:3].sum()],
                        df.iloc[num_pre_class[:3].sum()+ttc[3]:num_pre_class[:4].sum()],
                        df.iloc[num_pre_class[:4].sum()+ttc[4]:num_pre_class[:5].sum()],
                        df.iloc[num_pre_class[:5].sum()+ttc[5]:num_pre_class[:6].sum()],
                        df.iloc[num_pre_class[:6].sum()+ttc[6]:num_pre_class[:7].sum()],
                        df.iloc[num_pre_class[:7].sum()+ttc[7]:num_pre_class[:8].sum()],
                        df.iloc[num_pre_class[:8].sum()+ttc[8]:num_pre_class[:9].sum()],
                        df.iloc[num_pre_class[:9].sum()+ttc[9]:]))
y_test = np.concatenate(( np.zeros(num_pre_class[0]-ttc[0]) ,
                        np.ones(num_pre_class[1]-ttc[1]),
                        np.ones(num_pre_class[2]-ttc[2])*2,
                        np.ones(num_pre_class[3]-ttc[3])*3,
                        np.ones(num_pre_class[4]-ttc[4])*4,
                        np.ones(num_pre_class[5]-ttc[5])*5,
                        np.ones(num_pre_class[6]-ttc[6])*6,
                        np.ones(num_pre_class[7]-ttc[7])*7,
                        np.ones(num_pre_class[8]-ttc[8])*8,
                        np.ones(num_pre_class[9]-ttc[9])*9,))


#print(X_train.shape)
#print(y_train.shape)
#print(X_test.shape)
#print(y_test.shape)

idx = np.random.permutation(len(X_train))
X_train = X_train[idx]
y_train = y_train[idx]
idx = np.random.permutation(len(X_test))
X_test = X_test[idx]
y_test = y_test[idx]

from tensorflow import keras

y_train_encode = keras.utils.to_categorical(y_train)
y_test_encode = keras.utils.to_categorical(y_test)

def nnmodel(input_shape):
    X_input = keras.layers.Input((input_shape))
    #X = keras.layers.Dense(1024,activation='relu')(X_input)
    X = keras.layers.Dense(2048)(X_input)
    X = keras.layers.LeakyReLU(alpha=0.3)(X)
    X = keras.layers.Dense(512)(X)
    X = keras.layers.LeakyReLU(alpha=0.3)(X)
    X = keras.layers.Dense(256)(X)
    X = keras.layers.LeakyReLU(alpha=0.3)(X)
    X = keras.layers.Dense(10,activation='softmax')(X)
    model = keras.models.Model(inputs=X_input, outputs=X, name='model')
    return model


mymodel = nnmodel(X_train[0].shape)
mymodel.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['accuracy'])
mymodel.summary()

mymodel.fit(X_train,y_train_encode,batch_size=32,epochs=20,validation_data=(X_test[:200],y_test_encode[:200]))

from sklearn.metrics import confusion_matrix,cohen_kappa_score,classification_report

#print(mymodel.evaluate(X_test,y_test_encode))

pred = mymodel.predict(X_test)

print(classification_report(y_test,np.argmax(pred,axis=1)))

conMat = confusion_matrix(y_test,np.argmax(pred,axis=1))
print('confusion_matrix:')
print(conMat)

overall_acc = (conMat*np.eye(len(conMat))).sum()/conMat.sum()
print(f'Overall accuracy : {overall_acc}')


kappa = cohen_kappa_score(y_test,np.argmax(pred,axis=1))

print(f'kappa score : {kappa}')

mymodel.save('final5_model.h5')