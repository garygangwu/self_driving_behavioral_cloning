import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, Merge, Lambda, Cropping2D
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from training_data_loader import *

model_file = 'model.h5'

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((50,20), (0,0))))
model.add(Conv2D(24, (5, 5), activation='elu', subsample=(2,2)))
model.add(Conv2D(36, (5, 5), activation='elu', subsample=(2,2)))
model.add(Conv2D(48, (5, 5), activation='elu', subsample=(2,2)))
model.add(Conv2D(64, (3, 3), activation='elu'))
model.add(Conv2D(64, (3, 3), activation='elu'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100, activation='elu'))
model.add(Dense(50, activation='elu'))
model.add(Dense(10, activation='elu'))
model.add(Dense(1))
model.summary()

checkpoint = ModelCheckpoint('model-{epoch:03d}.h5', monitor='val_loss', verbose=0, save_best_only=True, mode='auto')
model.compile(loss='mean_squared_error', optimizer='adam')

X_train, y_train = load_data()

model.fit(X_train, y_train,
          epochs=10, batch_size=128, validation_split=0.2, shuffle=True,
          callbacks=[checkpoint])

model.save(model_file)

l = model.predict(X_train, batch_size=100)
print(l)
