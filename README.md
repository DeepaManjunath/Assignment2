# Assignment2
EIP4 - Assignment2













import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Add, BatchNormalization
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils

from keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()


print (X_train.shape)
from matplotlib import pyplot as plt
%matplotlib inline
plt.imshow(X_train[0])

X_train = X_train.reshape(X_train.shape[0], 28, 28,1)
X_test = X_test.reshape(X_test.shape[0], 28, 28,1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

y_train[:10]
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

Y_train[:10]

from keras.layers import Activation
model = Sequential()
 
model.add(Convolution2D(8, 3, 3, activation='relu', input_shape=(28,28,1),bias=False)) #26
model.add(BatchNormalization())
model.add(Dropout(0.13))

model.add(Convolution2D(8, 3, 3, activation='relu',bias=False)) #24
model.add(BatchNormalization())
model.add(Dropout(0.13))

model.add(Convolution2D(8,3,3,activation='relu',bias=False))#22
model.add(BatchNormalization())
model.add(Dropout(0.13))

model.add(MaxPooling2D(pool_size=(2,2)))#11

model.add(Convolution2D(8, 1, 1, activation='relu',bias=False)) #11



model.add(Convolution2D(10, 3, 3, activation='relu',bias=False))#9
model.add(BatchNormalization())
model.add(Dropout(0.13))


model.add(Convolution2D(10, 3, 3, activation='relu',bias=False))#7
model.add(BatchNormalization())
model.add(Dropout(0.13))


model.add(Convolution2D(10, 3, 3, activation='relu',bias=False))#5
model.add(BatchNormalization())
model.add(Dropout(0.13))


model.add(Convolution2D(10, 3, 3, activation='relu',bias=False))#3
model.add(BatchNormalization())
model.add(Dropout(0.13))


model.add(Convolution2D(10, 3, 3,bias=False))



model.add(Flatten())
model.add(Activation('softmax'))


model.summary()












import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Add, BatchNormalization
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils

from keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()


print (X_train.shape)
from matplotlib import pyplot as plt
%matplotlib inline
plt.imshow(X_train[0])

X_train = X_train.reshape(X_train.shape[0], 28, 28,1)
X_test = X_test.reshape(X_test.shape[0], 28, 28,1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

y_train[:10]
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

Y_train[:10]

from keras.layers import Activation
model = Sequential()
 
model.add(Convolution2D(8, 3, 3, activation='relu', input_shape=(28,28,1),bias=False)) #26
model.add(BatchNormalization())
model.add(Dropout(0.13))

model.add(Convolution2D(8, 3, 3, activation='relu',bias=False)) #24
model.add(BatchNormalization())
model.add(Dropout(0.13))

model.add(Convolution2D(8,3,3,activation='relu',bias=False))#22
model.add(BatchNormalization())
model.add(Dropout(0.13))

model.add(MaxPooling2D(pool_size=(2,2)))#11

model.add(Convolution2D(8, 1, 1, activation='relu',bias=False)) #11



model.add(Convolution2D(10, 3, 3, activation='relu',bias=False))#9
model.add(BatchNormalization())
model.add(Dropout(0.13))


model.add(Convolution2D(10, 3, 3, activation='relu',bias=False))#7
model.add(BatchNormalization())
model.add(Dropout(0.13))


model.add(Convolution2D(10, 3, 3, activation='relu',bias=False))#5
model.add(BatchNormalization())
model.add(Dropout(0.13))


model.add(Convolution2D(10, 3, 3, activation='relu',bias=False))#3
model.add(BatchNormalization())
model.add(Dropout(0.13))


model.add(Convolution2D(10, 3, 3,bias=False))



model.add(Flatten())
model.add(Activation('softmax'))


model.summary()
Model: "sequential_3"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_19 (Conv2D)           (None, 26, 26, 8)         72        
_________________________________________________________________
batch_normalization_15 (Batc (None, 26, 26, 8)         32        
_________________________________________________________________
dropout_15 (Dropout)         (None, 26, 26, 8)         0         
_________________________________________________________________
conv2d_20 (Conv2D)           (None, 24, 24, 8)         576       
_________________________________________________________________
batch_normalization_16 (Batc (None, 24, 24, 8)         32        
_________________________________________________________________
dropout_16 (Dropout)         (None, 24, 24, 8)         0         
_________________________________________________________________
conv2d_21 (Conv2D)           (None, 22, 22, 8)         576       
_________________________________________________________________
batch_normalization_17 (Batc (None, 22, 22, 8)         32        
_________________________________________________________________
dropout_17 (Dropout)         (None, 22, 22, 8)         0         
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 11, 11, 8)         0         
_________________________________________________________________
conv2d_22 (Conv2D)           (None, 11, 11, 8)         64        
_________________________________________________________________
conv2d_23 (Conv2D)           (None, 9, 9, 10)          720       
_________________________________________________________________
batch_normalization_18 (Batc (None, 9, 9, 10)          40        
_________________________________________________________________
dropout_18 (Dropout)         (None, 9, 9, 10)          0         
_________________________________________________________________
conv2d_24 (Conv2D)           (None, 7, 7, 10)          900       
_________________________________________________________________
batch_normalization_19 (Batc (None, 7, 7, 10)          40        
_________________________________________________________________
dropout_19 (Dropout)         (None, 7, 7, 10)          0         
_________________________________________________________________
conv2d_25 (Conv2D)           (None, 5, 5, 10)          900       
_________________________________________________________________
batch_normalization_20 (Batc (None, 5, 5, 10)          40        
_________________________________________________________________
dropout_20 (Dropout)         (None, 5, 5, 10)          0         
_________________________________________________________________
conv2d_26 (Conv2D)           (None, 3, 3, 10)          900       
_________________________________________________________________
batch_normalization_21 (Batc (None, 3, 3, 10)          40        
_________________________________________________________________
dropout_21 (Dropout)         (None, 3, 3, 10)          0         
_________________________________________________________________
conv2d_27 (Conv2D)           (None, 1, 1, 10)          900       
_________________________________________________________________
flatten_3 (Flatten)          (None, 10)                0         
_________________________________________________________________
activation_3 (Activation)    (None, 10)                0         
=================================================================
Total params: 5,864
Trainable params: 5,736
Non-trainable params: 128
________________________________________________

from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler
def scheduler(epoch, lr):
  return round(0.00994 * 1/(1 + 0.319 * epoch), 10)

model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.003), metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=128, epochs=20, verbose=1, validation_data=(X_test, Y_test), callbacks=[LearningRateScheduler(scheduler, verbose=1)])
Epoch 1/20

Epoch 00001: LearningRateScheduler setting learning rate to 0.00994.
60000/60000 [==============================] - 96s 2ms/step - loss: 0.0588 - acc: 0.9818 - val_loss: 0.0360 - val_acc: 0.9893
Epoch 2/20

Epoch 00002: LearningRateScheduler setting learning rate to 0.0075360121.
60000/60000 [==============================] - 91s 2ms/step - loss: 0.0509 - acc: 0.9842 - val_loss: 0.0361 - val_acc: 0.9885
Epoch 3/20

Epoch 00003: LearningRateScheduler setting learning rate to 0.0060683761.
60000/60000 [==============================] - 92s 2ms/step - loss: 0.0471 - acc: 0.9853 - val_loss: 0.0325 - val_acc: 0.9908
Epoch 4/20

Epoch 00004: LearningRateScheduler setting learning rate to 0.0050792029.
60000/60000 [==============================] - 91s 2ms/step - loss: 0.0434 - acc: 0.9862 - val_loss: 0.0279 - val_acc: 0.9917
Epoch 5/20

Epoch 00005: LearningRateScheduler setting learning rate to 0.0043673111.
60000/60000 [==============================] - 88s 1ms/step - loss: 0.0411 - acc: 0.9869 - val_loss: 0.0286 - val_acc: 0.9915
Epoch 6/20

Epoch 00006: LearningRateScheduler setting learning rate to 0.0038304432.
60000/60000 [==============================] - 93s 2ms/step - loss: 0.0373 - acc: 0.9882 - val_loss: 0.0247 - val_acc: 0.9928
Epoch 7/20

Epoch 00007: LearningRateScheduler setting learning rate to 0.0034111187.
60000/60000 [==============================] - 94s 2ms/step - loss: 0.0364 - acc: 0.9885 - val_loss: 0.0285 - val_acc: 0.9916
Epoch 8/20

Epoch 00008: LearningRateScheduler setting learning rate to 0.0030745438.
60000/60000 [==============================] - 89s 1ms/step - loss: 0.0373 - acc: 0.9882 - val_loss: 0.0266 - val_acc: 0.9925
Epoch 9/20

Epoch 00009: LearningRateScheduler setting learning rate to 0.0027984234.
60000/60000 [==============================] - 91s 2ms/step - loss: 0.0366 - acc: 0.9883 - val_loss: 0.0240 - val_acc: 0.9923
Epoch 10/20

Epoch 00010: LearningRateScheduler setting learning rate to 0.0025678119.
60000/60000 [==============================] - 93s 2ms/step - loss: 0.0340 - acc: 0.9891 - val_loss: 0.0271 - val_acc: 0.9921
Epoch 11/20

Epoch 00011: LearningRateScheduler setting learning rate to 0.002372315.
60000/60000 [==============================] - 92s 2ms/step - loss: 0.0352 - acc: 0.9885 - val_loss: 0.0227 - val_acc: 0.9931
Epoch 12/20

Epoch 00012: LearningRateScheduler setting learning rate to 0.0022044799.
60000/60000 [==============================] - 94s 2ms/step - loss: 0.0334 - acc: 0.9890 - val_loss: 0.0216 - val_acc: 0.9937
Epoch 13/20

Epoch 00013: LearningRateScheduler setting learning rate to 0.0020588235.
60000/60000 [==============================] - 94s 2ms/step - loss: 0.0332 - acc: 0.9896 - val_loss: 0.0254 - val_acc: 0.9929
Epoch 14/20

Epoch 00014: LearningRateScheduler setting learning rate to 0.0019312221.
60000/60000 [==============================] - 91s 2ms/step - loss: 0.0334 - acc: 0.9894 - val_loss: 0.0226 - val_acc: 0.9943
Epoch 15/20

Epoch 00015: LearningRateScheduler setting learning rate to 0.0018185145.
60000/60000 [==============================] - 93s 2ms/step - loss: 0.0319 - acc: 0.9898 - val_loss: 0.0239 - val_acc: 0.9931
Epoch 16/20

Epoch 00016: LearningRateScheduler setting learning rate to 0.0017182368.
60000/60000 [==============================] - 94s 2ms/step - loss: 0.0317 - acc: 0.9898 - val_loss: 0.0197 - val_acc: 0.9942
Epoch 17/20

Epoch 00017: LearningRateScheduler setting learning rate to 0.0016284404.
60000/60000 [==============================] - 94s 2ms/step - loss: 0.0322 - acc: 0.9898 - val_loss: 0.0213 - val_acc: 0.9939
Epoch 18/20

Epoch 00018: LearningRateScheduler setting learning rate to 0.0015475634.
60000/60000 [==============================] - 93s 2ms/step - loss: 0.0305 - acc: 0.9901 - val_loss: 0.0205 - val_acc: 0.9936
Epoch 19/20

Epoch 00019: LearningRateScheduler setting learning rate to 0.00147434.
60000/60000 [==============================] - 94s 2ms/step - loss: 0.0306 - acc: 0.9904 - val_loss: 0.0241 - val_acc: 0.9926
Epoch 20/20

Epoch 00020: LearningRateScheduler setting learning rate to 0.0014077326.
60000/60000 [==============================] - 93s 2ms/step - loss: 0.0312 - acc: 0.9901 - val_loss: 0.0203 - val_acc: 0.9935
<keras.callbacks.History at 0x7f52ae8cdc50>


score = model.evaluate(X_test, Y_test, verbose=0)
print(score)



from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler
def scheduler(epoch, lr):
  return round(0.00994 * 1/(1 + 0.319 * epoch), 10)

model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.003), metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=128, epochs=20, verbose=1, validation_data=(X_test, Y_test), callbacks=[LearningRateScheduler(scheduler, verbose=1)])


score = model.evaluate(X_test, Y_test, verbose=0)
print(score)
[0.02025903364851256, 0.9935]

Strategy to get the Results:

Reduced  the number of  kernels to reduce the parameters
Have added batch normalization and dropout to increase the performance.
Convolution 1x1 after Maxpooling to reduce computations and to reduce number of parameters
Have used bias as false to avoid bias
Not added dropout before output.Tried to optimize the dropout number.
Tried to optimize the learning rate..and have set to 0.0087
Not added batch normalization before output.
===============================================================
28x28x1 is the input shape, have used 3x3,  8 kernels to get 26x26x8
Again used 3x3 , 8 kernels with activation fuction ‘Relu’ to get 24x24x8
again used 3x3 to get 22x22x8
Have used Maxpooling with 2x2 kernel to get 11X11x8
Have used 1x1 kernel for merging to get 11x11x8
And again used 3x3 kernel to get 9x9x10
And 3x3 kernel to get 7x7x10 
And 3x3 kernel to get 5x5x10
And 3x3 kernel to get 3x3x10
And have used larger 3x3 kernel in the end
And then have used softmax function for better output








import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Add, BatchNormalization
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils

from keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()


print (X_train.shape)
from matplotlib import pyplot as plt
%matplotlib inline
plt.imshow(X_train[0])

X_train = X_train.reshape(X_train.shape[0], 28, 28,1)
X_test = X_test.reshape(X_test.shape[0], 28, 28,1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

y_train[:10]
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

Y_train[:10]

from keras.layers import Activation
model = Sequential()
 
model.add(Convolution2D(8, 3, 3, activation='relu', input_shape=(28,28,1),bias=False)) #26
model.add(BatchNormalization())
model.add(Dropout(0.098))

model.add(Convolution2D(8, 3, 3, activation='relu',bias=False)) #24
model.add(BatchNormalization())
model.add(Dropout(0.098))

model.add(Convolution2D(8,3,3,activation='relu',bias=False))#22
model.add(BatchNormalization())
model.add(Dropout(0.098))

model.add(MaxPooling2D(pool_size=(2,2)))#11

model.add(Convolution2D(8, 1, 1, activation='relu',bias=False)) #11



model.add(Convolution2D(10, 3, 3, activation='relu',bias=False))#9
model.add(BatchNormalization())
model.add(Dropout(0.098))


model.add(Convolution2D(10, 3, 3, activation='relu',bias=False))#7
model.add(BatchNormalization())
model.add(Dropout(0.092))


model.add(Convolution2D(10, 3, 3, activation='relu',bias=False))#5
model.add(BatchNormalization())
model.add(Dropout(0.098))


model.add(Convolution2D(10, 3, 3, activation='relu',bias=False))#3
model.add(BatchNormalization())
model.add(Dropout(0.098))


model.add(Convolution2D(10, 3, 3,bias=False))
model.add(BatchNormalization())



model.add(Flatten())
model.add(Activation('softmax'))


model.summary()
Model: "sequential_6"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_46 (Conv2D)           (None, 26, 26, 8)         72        
_________________________________________________________________
batch_normalization_41 (Batc (None, 26, 26, 8)         32        
_________________________________________________________________
dropout_36 (Dropout)         (None, 26, 26, 8)         0         
_________________________________________________________________
conv2d_47 (Conv2D)           (None, 24, 24, 8)         576       
_________________________________________________________________
batch_normalization_42 (Batc (None, 24, 24, 8)         32        
_________________________________________________________________
dropout_37 (Dropout)         (None, 24, 24, 8)         0         
_________________________________________________________________
conv2d_48 (Conv2D)           (None, 22, 22, 8)         576       
_________________________________________________________________
batch_normalization_43 (Batc (None, 22, 22, 8)         32        
_________________________________________________________________
dropout_38 (Dropout)         (None, 22, 22, 8)         0         
_________________________________________________________________
max_pooling2d_6 (MaxPooling2 (None, 11, 11, 8)         0         
_________________________________________________________________
conv2d_49 (Conv2D)           (None, 11, 11, 8)         64        
_________________________________________________________________
conv2d_50 (Conv2D)           (None, 9, 9, 10)          720       
_________________________________________________________________
batch_normalization_44 (Batc (None, 9, 9, 10)          40        
_________________________________________________________________
dropout_39 (Dropout)         (None, 9, 9, 10)          0         
_________________________________________________________________
conv2d_51 (Conv2D)           (None, 7, 7, 10)          900       
_________________________________________________________________
batch_normalization_45 (Batc (None, 7, 7, 10)          40        
_________________________________________________________________
dropout_40 (Dropout)         (None, 7, 7, 10)          0         
_________________________________________________________________
conv2d_52 (Conv2D)           (None, 5, 5, 10)          900       
_________________________________________________________________
batch_normalization_46 (Batc (None, 5, 5, 10)          40        
_________________________________________________________________
dropout_41 (Dropout)         (None, 5, 5, 10)          0         
_________________________________________________________________
conv2d_53 (Conv2D)           (None, 3, 3, 10)          900       
_________________________________________________________________
batch_normalization_47 (Batc (None, 3, 3, 10)          40        
_________________________________________________________________
dropout_42 (Dropout)         (None, 3, 3, 10)          0         
_________________________________________________________________
conv2d_54 (Conv2D)           (None, 1, 1, 10)          900       
_________________________________________________________________
batch_normalization_48 (Batc (None, 1, 1, 10)          40        
_________________________________________________________________
flatten_6 (Flatten)          (None, 10)                0         
_________________________________________________________________
activation_6 (Activation)    (None, 10)                0         
=================================================================
Total params: 5,904
Trainable params: 5,756
Non-trainable params: 148
_____

from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler
def scheduler(epoch, lr):
  return round(0.0087 * 1/(1 + 0.319 * epoch), 10)

model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.003), metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=128, epochs=20, verbose=1, validation_data=(X_test, Y_test), callbacks=[LearningRateScheduler(scheduler, verbose=1)])
Train on 60000 samples, validate on 10000 samples
Epoch 1/20

Epoch 00001: LearningRateScheduler setting learning rate to 0.0087.
60000/60000 [==============================] - 87s 1ms/step - loss: 0.3206 - acc: 0.9238 - val_loss: 0.0654 - val_acc: 0.9801
Epoch 2/20

Epoch 00002: LearningRateScheduler setting learning rate to 0.006595906.
60000/60000 [==============================] - 83s 1ms/step - loss: 0.0971 - acc: 0.9737 - val_loss: 0.0650 - val_acc: 0.9792
Epoch 3/20

Epoch 00003: LearningRateScheduler setting learning rate to 0.0053113553.
60000/60000 [==============================] - 84s 1ms/step - loss: 0.0770 - acc: 0.9779 - val_loss: 0.0401 - val_acc: 0.9886
Epoch 4/20

Epoch 00004: LearningRateScheduler setting learning rate to 0.00444558.
60000/60000 [==============================] - 84s 1ms/step - loss: 0.0663 - acc: 0.9806 - val_loss: 0.0335 - val_acc: 0.9890
Epoch 5/20

Epoch 00005: LearningRateScheduler setting learning rate to 0.0038224956.
60000/60000 [==============================] - 84s 1ms/step - loss: 0.0578 - acc: 0.9830 - val_loss: 0.0354 - val_acc: 0.9898
Epoch 6/20

Epoch 00006: LearningRateScheduler setting learning rate to 0.0033526012.
60000/60000 [==============================] - 85s 1ms/step - loss: 0.0550 - acc: 0.9837 - val_loss: 0.0310 - val_acc: 0.9909
Epoch 7/20

Epoch 00007: LearningRateScheduler setting learning rate to 0.0029855868.
60000/60000 [==============================] - 84s 1ms/step - loss: 0.0530 - acc: 0.9845 - val_loss: 0.0280 - val_acc: 0.9913
Epoch 8/20

Epoch 00008: LearningRateScheduler setting learning rate to 0.0026909991.
60000/60000 [==============================] - 84s 1ms/step - loss: 0.0507 - acc: 0.9850 - val_loss: 0.0267 - val_acc: 0.9925
Epoch 9/20

Epoch 00009: LearningRateScheduler setting learning rate to 0.0024493243.
60000/60000 [==============================] - 85s 1ms/step - loss: 0.0480 - acc: 0.9862 - val_loss: 0.0297 - val_acc: 0.9921
Epoch 10/20

Epoch 00010: LearningRateScheduler setting learning rate to 0.0022474813.
60000/60000 [==============================] - 85s 1ms/step - loss: 0.0449 - acc: 0.9862 - val_loss: 0.0254 - val_acc: 0.9918
Epoch 11/20

Epoch 00011: LearningRateScheduler setting learning rate to 0.0020763723.
60000/60000 [==============================] - 85s 1ms/step - loss: 0.0433 - acc: 0.9877 - val_loss: 0.0239 - val_acc: 0.9929
Epoch 12/20

Epoch 00012: LearningRateScheduler setting learning rate to 0.0019294744.
60000/60000 [==============================] - 84s 1ms/step - loss: 0.0423 - acc: 0.9870 - val_loss: 0.0239 - val_acc: 0.9937
Epoch 13/20

Epoch 00013: LearningRateScheduler setting learning rate to 0.0018019884.
60000/60000 [==============================] - 85s 1ms/step - loss: 0.0412 - acc: 0.9879 - val_loss: 0.0223 - val_acc: 0.9932
Epoch 14/20

Epoch 00014: LearningRateScheduler setting learning rate to 0.001690305.
60000/60000 [==============================] - 84s 1ms/step - loss: 0.0412 - acc: 0.9874 - val_loss: 0.0237 - val_acc: 0.9924
Epoch 15/20

Epoch 00015: LearningRateScheduler setting learning rate to 0.0015916575.
60000/60000 [==============================] - 84s 1ms/step - loss: 0.0391 - acc: 0.9882 - val_loss: 0.0287 - val_acc: 0.9909
Epoch 16/20

Epoch 00016: LearningRateScheduler setting learning rate to 0.0015038894.
60000/60000 [==============================] - 83s 1ms/step - loss: 0.0378 - acc: 0.9887 - val_loss: 0.0246 - val_acc: 0.9927
Epoch 17/20

Epoch 00017: LearningRateScheduler setting learning rate to 0.0014252949.
60000/60000 [==============================] - 83s 1ms/step - loss: 0.0364 - acc: 0.9892 - val_loss: 0.0245 - val_acc: 0.9936
Epoch 18/20

Epoch 00018: LearningRateScheduler setting learning rate to 0.0013545072.
60000/60000 [==============================] - 85s 1ms/step - loss: 0.0373 - acc: 0.9886 - val_loss: 0.0226 - val_acc: 0.9928
Epoch 19/20

Epoch 00019: LearningRateScheduler setting learning rate to 0.0012904183.
60000/60000 [==============================] - 85s 1ms/step - loss: 0.0351 - acc: 0.9896 - val_loss: 0.0221 - val_acc: 0.9946
Epoch 20/20

Epoch 00020: LearningRateScheduler setting learning rate to 0.0012321201.
60000/60000 [==============================] - 85s 1ms/step - loss: 0.0349 - acc: 0.9892 - val_loss: 0.0209 - val_acc: 0.9943
<keras.callbacks.History at 0x7faf0ecb5da0>

score = model.evaluate(X_test, Y_test, verbose=0)
print(score)
[0.020923026478011163, 0.9943]

Strategy to get the Results:

Reduced  the number of  kernels to reduce the parameters
Have added batch normalization and dropout to increase the performance.
Convolution 1x1 after Maxpooling to reduce computations and to reduce number of parameters
Have used bias as false to avoid bias
Not added dropout before output.Tried to optimize the dropout number.
Tried to optimize the learning rate..and have set to 0.0087
===============================================================
28x28x1 is the input shape, have used 3x3,  8 kernels to get 26x26x8
Again used 3x3 , 8 kernels with activation fuction ‘Relu’ to get 24x24x8
again used 3x3 to get 22x22x8
Have used Maxpooling with 2x2 kernel to get 11X11x8
Have used 1x1 kernel for merging to get 11x11x8
And again used 3x3 kernel to get 9x9x10
And 3x3 kernel to get 7x7x10 
And 3x3 kernel to get 5x5x10
And 3x3 kernel to get 3x3x10
And have used larger 3x3 kernel in the end
And then have used softmax function for better output





=================================================================================================================================================
=====
