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
 
model.add(Convolution2D(10, 3, 3, activation='relu', input_shape=(28,28,1),bias=False)) #26
model.add(BatchNormalization())
model.add(Dropout(0.1))

model.add(Convolution2D(10, 3, 3, activation='relu',bias=False)) #24
model.add(BatchNormalization())
model.add(Dropout(0.1))

model.add(MaxPooling2D(pool_size=(2,2)))#12

model.add(Convolution2D(10, 1, 1, activation='relu',bias=False)) #12



model.add(Convolution2D(10, 3, 3, activation='relu',bias=False))#10
model.add(BatchNormalization())
model.add(Dropout(0.1))


model.add(Convolution2D(10, 3, 3, activation='relu',bias=False))#8
model.add(BatchNormalization())
model.add(Dropout(0.1))


model.add(Convolution2D(10, 3, 3, activation='relu',bias=False))#6
model.add(BatchNormalization())
model.add(Dropout(0.1))


model.add(Convolution2D(10, 3, 3, activation='relu',bias=False))#4
model.add(BatchNormalization())
model.add(Dropout(0.1))


model.add(Convolution2D(10, 4, 4,bias=False))
model.add(BatchNormalization())



model.add(Flatten())
model.add(Activation('softmax'))


model.summary()
Model: "sequential_7"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_47 (Conv2D)           (None, 26, 26, 10)        90        
_________________________________________________________________
batch_normalization_38 (Batc (None, 26, 26, 10)        40        
_________________________________________________________________
dropout_37 (Dropout)         (None, 26, 26, 10)        0         
_________________________________________________________________
conv2d_48 (Conv2D)           (None, 24, 24, 10)        900       
_________________________________________________________________
batch_normalization_39 (Batc (None, 24, 24, 10)        40        
_________________________________________________________________
dropout_38 (Dropout)         (None, 24, 24, 10)        0         
_________________________________________________________________
max_pooling2d_7 (MaxPooling2 (None, 12, 12, 10)        0         
_________________________________________________________________
conv2d_49 (Conv2D)           (None, 12, 12, 10)        100       
_________________________________________________________________
conv2d_50 (Conv2D)           (None, 10, 10, 10)        900       
_________________________________________________________________
batch_normalization_40 (Batc (None, 10, 10, 10)        40        
_________________________________________________________________
dropout_39 (Dropout)         (None, 10, 10, 10)        0         
_________________________________________________________________
conv2d_51 (Conv2D)           (None, 8, 8, 10)          900       
_________________________________________________________________
batch_normalization_41 (Batc (None, 8, 8, 10)          40        
_________________________________________________________________
dropout_40 (Dropout)         (None, 8, 8, 10)          0         
_________________________________________________________________
conv2d_52 (Conv2D)           (None, 6, 6, 10)          900       
_________________________________________________________________
batch_normalization_42 (Batc (None, 6, 6, 10)          40        
_________________________________________________________________
dropout_41 (Dropout)         (None, 6, 6, 10)          0         
_________________________________________________________________
conv2d_53 (Conv2D)           (None, 4, 4, 10)          900       
_________________________________________________________________
batch_normalization_43 (Batc (None, 4, 4, 10)          40        
_________________________________________________________________
dropout_42 (Dropout)         (None, 4, 4, 10)          0         
_________________________________________________________________
conv2d_54 (Conv2D)           (None, 1, 1, 10)          1600      
_________________________________________________________________
batch_normalization_44 (Batc (None, 1, 1, 10)          40        
_________________________________________________________________
flatten_7 (Flatten)          (None, 10)                0         
_________________________________________________________________
activation_7 (Activation)    (None, 10)                0         
=================================================================
Total params: 6,570
Trainable params: 6,430
Non-trainable params: 140
_____

from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler
def scheduler(epoch, lr):
  return round(0.0085 * 1/(1 + 0.319 * epoch), 10)

model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.003), metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=128, epochs=20, verbose=1, validation_data=(X_test, Y_test), callbacks=[LearningRateScheduler(scheduler, verbose=1)])

Train on 60000 samples, validate on 10000 samples
Epoch 1/20

Epoch 00001: LearningRateScheduler setting learning rate to 0.0085.
60000/60000 [==============================] - 87s 1ms/step - loss: 0.0727 - acc: 0.9783 - val_loss: 0.0495 - val_acc: 0.9843
Epoch 2/20

Epoch 00002: LearningRateScheduler setting learning rate to 0.006444276.
60000/60000 [==============================] - 85s 1ms/step - loss: 0.0598 - acc: 0.9818 - val_loss: 0.0582 - val_acc: 0.9828
Epoch 3/20

Epoch 00003: LearningRateScheduler setting learning rate to 0.0051892552.
60000/60000 [==============================] - 84s 1ms/step - loss: 0.0515 - acc: 0.9840 - val_loss: 0.0295 - val_acc: 0.9913
Epoch 4/20

Epoch 00004: LearningRateScheduler setting learning rate to 0.0043433827.
60000/60000 [==============================] - 84s 1ms/step - loss: 0.0464 - acc: 0.9860 - val_loss: 0.0276 - val_acc: 0.9907
Epoch 5/20

Epoch 00005: LearningRateScheduler setting learning rate to 0.0037346221.
60000/60000 [==============================] - 84s 1ms/step - loss: 0.0438 - acc: 0.9870 - val_loss: 0.0279 - val_acc: 0.9908
Epoch 6/20

Epoch 00006: LearningRateScheduler setting learning rate to 0.0032755299.
60000/60000 [==============================] - 84s 1ms/step - loss: 0.0400 - acc: 0.9879 - val_loss: 0.0323 - val_acc: 0.9901
Epoch 7/20

Epoch 00007: LearningRateScheduler setting learning rate to 0.0029169526.
60000/60000 [==============================] - 85s 1ms/step - loss: 0.0383 - acc: 0.9884 - val_loss: 0.0325 - val_acc: 0.9900
Epoch 8/20

Epoch 00008: LearningRateScheduler setting learning rate to 0.002629137.
60000/60000 [==============================] - 86s 1ms/step - loss: 0.0367 - acc: 0.9888 - val_loss: 0.0259 - val_acc: 0.9915
Epoch 9/20

Epoch 00009: LearningRateScheduler setting learning rate to 0.002393018.
60000/60000 [==============================] - 85s 1ms/step - loss: 0.0351 - acc: 0.9896 - val_loss: 0.0245 - val_acc: 0.9922
Epoch 10/20

Epoch 00010: LearningRateScheduler setting learning rate to 0.002195815.
60000/60000 [==============================] - 85s 1ms/step - loss: 0.0347 - acc: 0.9894 - val_loss: 0.0237 - val_acc: 0.9921
Epoch 11/20

Epoch 00011: LearningRateScheduler setting learning rate to 0.0020286396.
60000/60000 [==============================] - 85s 1ms/step - loss: 0.0347 - acc: 0.9893 - val_loss: 0.0246 - val_acc: 0.9921
Epoch 12/20

Epoch 00012: LearningRateScheduler setting learning rate to 0.0018851187.
60000/60000 [==============================] - 85s 1ms/step - loss: 0.0327 - acc: 0.9901 - val_loss: 0.0259 - val_acc: 0.9922
Epoch 13/20

Epoch 00013: LearningRateScheduler setting learning rate to 0.0017605634.
60000/60000 [==============================] - 85s 1ms/step - loss: 0.0312 - acc: 0.9903 - val_loss: 0.0200 - val_acc: 0.9940
Epoch 14/20

Epoch 00014: LearningRateScheduler setting learning rate to 0.0016514474.
60000/60000 [==============================] - 85s 1ms/step - loss: 0.0304 - acc: 0.9904 - val_loss: 0.0230 - val_acc: 0.9922
Epoch 15/20

Epoch 00015: LearningRateScheduler setting learning rate to 0.0015550677.
60000/60000 [==============================] - 85s 1ms/step - loss: 0.0304 - acc: 0.9906 - val_loss: 0.0232 - val_acc: 0.9927
Epoch 16/20

Epoch 00016: LearningRateScheduler setting learning rate to 0.0014693172.
60000/60000 [==============================] - 86s 1ms/step - loss: 0.0295 - acc: 0.9907 - val_loss: 0.0223 - val_acc: 0.9932
Epoch 17/20

Epoch 00017: LearningRateScheduler setting learning rate to 0.0013925295.
60000/60000 [==============================] - 86s 1ms/step - loss: 0.0301 - acc: 0.9908 - val_loss: 0.0214 - val_acc: 0.9935
Epoch 18/20

Epoch 00018: LearningRateScheduler setting learning rate to 0.0013233691.
60000/60000 [==============================] - 85s 1ms/step - loss: 0.0282 - acc: 0.9913 - val_loss: 0.0219 - val_acc: 0.9932
Epoch 19/20

Epoch 00019: LearningRateScheduler setting learning rate to 0.0012607535.
60000/60000 [==============================] - 85s 1ms/step - loss: 0.0286 - acc: 0.9911 - val_loss: 0.0208 - val_acc: 0.9937
Epoch 20/20

Epoch 00020: LearningRateScheduler setting learning rate to 0.0012037955.
60000/60000 [==============================] - 85s 1ms/step - loss: 0.0291 - acc: 0.9908 - val_loss: 0.0206 - val_acc: 0.9940
<keras.callbacks.History at 0x7ffb4f8eb240>

score = model.evaluate(X_test, Y_test, verbose=0)
print(score)
[0.020602115210262128, 0.994]

Strategy to get the Results:

Reduced  the number of  kernels to reduce the parameters
Have added batch normalization and dropout to increase the performance.
Convolution 1x1 after Maxpooling to reduce computations and to reduce number of parameters
Have used bias as false to avoid bias
Not added dropout before output.
Tried to optimize the learning rate..and have set to 0.0085
===============================================================
28x28x1 is the input shape, have used 3x3,  10 kernels to get 26x26x10
Again used 3x3 , 10 kernels with activation fuction ‘Relu’ to get 24x24x10
Have used Maxpooling with 2x2 kernel to get 12X12x10
Have used 1x1 kernel for merging to get 12x12x10
And again used 3x3 kernel to get 10x10x10
And 3x3 kernel to get 8x8x10 
And 3x3 kernel to get 6x6x10
And 3x3 kernel to get 4x4x10
And have used larger 4x4 kernel in the end
And then have used softmax function for better output



