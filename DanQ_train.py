import numpy as np
import h5py
import scipy.io
np.random.seed(1337) # for reproducibility

from keras import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers.recurrent import LSTM, GRU
from keras.callbacks import ModelCheckpoint, EarlyStopping
# from seya.layers.recurrent import Bidirectional

from keras.layers import Bidirectional

print('#### Loading DATA ###########################################')
print("-- Loading Training Set")
trainmat = scipy.io.loadmat('data/train_trunc.mat') #FOR TRUNC: needed to change this to scipy.loadmat instead of h5py.File
print("-- Loading Validation Set")
validmat = scipy.io.loadmat('data/valid.mat')
print("-- Loading Test Set")
testmat = scipy.io.loadmat('data/test_trunc.mat')

print("### Dividing Training data into X_train and y_train #########")
print("-- setting up X_train")
X_train = np.transpose(np.array(trainmat['trainx_trunc']),axes=(0,2,1)) # FOR TRUNC: switched transpose axes from to (2, 0, 1) to (0, 2, 1)
print("-- setting up y_train")
y_train = np.array(trainmat['train_trunc']) # FOR TRUNC: remove the transpose

print("Setting up B-RNN Layer")
forward_lstm = LSTM(units=320, input_dim=320, return_sequences=True)
backward_lstm = LSTM(units=320, input_dim=320, return_sequences=True, go_backwards=True)
brnn = Bidirectional(layer=forward_lstm, backward_layer=backward_lstm)

print('### Building Model ##########################')

model = Sequential()
model.add(Conv1D(
                filters=320,
                kernel_size=26,
                input_shape = (1000,4),
                activation="relu",
                padding="valid",
                strides=1))

model.add(MaxPooling1D(pool_size=13, strides=13))

model.add(Dropout(0.2))

model.add(brnn)

model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(input_dim=75*640, units=925))
model.add(Activation('relu'))

model.add(Dense(input_dim=925, units=919))
model.add(Activation('sigmoid'))

print('compiling model')
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=["accuracy"])

print('running at most 60 epochs')

checkpointer = ModelCheckpoint(filepath="DanQ_bestmodel.hdf5", verbose=1, save_best_only=True)
earlystopper = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

model.fit(X_train, y_train, batch_size=100, epochs=60, shuffle=True,
        validation_data=(np.transpose(validmat['validxdata'],axes=(0,2,1)),
        validmat['validdata']), callbacks=[checkpointer,earlystopper]
    )

tresults = model.evaluate(np.transpose(testmat['testx_trunc'],axes=(0,2,1)), testmat['test_trunc'])

print(tresults)

