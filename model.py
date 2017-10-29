import csv
import matplotlib.image as mpimg
import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, Merge, Lambda, Cropping2D
from keras.utils import np_utils
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('epochs', 25, "The number of epochs.")
flags.DEFINE_integer('batch_size', 128, "The batch size.")
flags.DEFINE_string('road', 'hard', "Specify the simulator road")

training_file_config = {
  'easy': [
    {
      'log_file': 'data/driving_log.csv',
      'image_fold': 'data/IMG/'
    }
  ],
  'hard': [
    {
      'log_file': 'data_tough_roads/driving_log.csv',
      'image_fold': 'data_tough_roads/IMG/'
    },
    {
      'log_file': 'data_hard_additional/driving_log.csv',
      'image_fold': 'data_hard_additional/IMG/'
    },
    {
      'log_file': 'data_hard_additional_2/driving_log.csv',
      'image_fold': 'data_hard_additional_2/IMG/'
    }
  ]
}

def load_data():
  file_configs = training_file_config[FLAGS.road]
  records = []

  for config in file_configs:
    driving_log_file = config['log_file']
    image_fold = config['image_fold']
    reader = csv.reader(open(driving_log_file))
    for row in reader:
      r = {
        'center' : image_fold + row[0].split('/')[-1],
        'left'   : image_fold + row[1].split('/')[-1],
        'right'  : image_fold + row[2].split('/')[-1],
        'steering' : float(row[3]),
        'throttle' : float(row[4]),
        'brake'    : float(row[5]),
        'speed'    : float(row[6])
      }
      records.append(r)

  X = []
  y = []
  for record in records:
    center_img_file_name = record['center']
    left_img_file_name   = record['left']
    right_img_file_name  = record['right']
    steering = record['steering']
    throttle = record['throttle']
    brake    = record['brake']
    speed    = record['speed']

    img = mpimg.imread(center_img_file_name)
    X.append(img)
    y.append(steering)
    X.append(np.fliplr(img)) # flip the image for extra training data
    y.append(-steering)

    img = mpimg.imread(left_img_file_name)
    left_steering = steering + 0.25
    X.append(img)
    y.append(left_steering)
    X.append(np.fliplr(img)) # flip the image for extra training data
    y.append(-left_steering)

    img = mpimg.imread(right_img_file_name)
    right_steering = steering - 0.25
    X.append(img)
    y.append(right_steering)
    X.append(np.fliplr(img)) # flip the image for extra training data
    y.append(-right_steering)

  return np.array(X), np.array(y)


model_file = 'model.h5'

model = Sequential()
model.add(Lambda(lambda x: (x / 128.0) - 1.0, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((50,20), (0,0))))
model.add(Conv2D(24, (5, 5), activation='elu', strides=(2,2), kernel_regularizer = l2(0.001)))
model.add(Conv2D(36, (5, 5), activation='elu', strides=(2,2), kernel_regularizer = l2(0.001)))
model.add(Conv2D(48, (5, 5), activation='elu', strides=(2,2), kernel_regularizer = l2(0.001)))
model.add(Conv2D(64, (3, 3), activation='elu', kernel_regularizer = l2(0.001)))
model.add(Conv2D(64, (3, 3), activation='elu', kernel_regularizer = l2(0.001)))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(100, activation='elu', kernel_regularizer = l2(0.001)))
model.add(Dropout(0.5))
model.add(Dense(50, activation='elu', kernel_regularizer = l2(0.001)))
model.add(Dense(10, activation='elu', kernel_regularizer = l2(0.001)))
model.add(Dense(1))
model.summary()

checkpoint = ModelCheckpoint('model-{epoch:03d}.h5', monitor='val_loss', verbose=0, save_best_only=True, mode='auto')
model.compile(loss='mean_squared_error', optimizer='adam')

X_train, y_train = load_data()
print(len(X_train))
print(y_train)

model.fit(X_train, y_train,
          epochs=FLAGS.epochs, batch_size=FLAGS.batch_size, validation_split=0.2, shuffle=True,
          callbacks=[checkpoint])

model.save(model_file)

l = model.predict(X_train, batch_size=100)
print(l)
