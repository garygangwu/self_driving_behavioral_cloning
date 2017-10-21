import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, Merge
from keras.utils import np_utils
from training_data_loader import *

model_file = 'model.h'

# left_branch = Sequential()
# left_branch.add(Conv2D(32, (3, 3), activation='relu', input_shape=(160, 320, 3)))

# middle_branch = Sequential()
# middle_branch.add(Conv2D(32, (3, 3), activation='relu', input_shape=(160, 320, 3)))

# right_branch = Sequential()
# right_branch.add(Conv2D(32, (3, 3), activation='relu', input_shape=(160, 320, 3)))

# merged = Merge([left_branch, middle_branch, right_branch], mode='concat')

# final_model = Sequential()
# final_model.add(merged)
# final_model.add(Dense(10, activation='softmax'))

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(160, 320, 3)))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('relu'))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')

X_train, y_train = load_data()
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
model.save(model_file)

