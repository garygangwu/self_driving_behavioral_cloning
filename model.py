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
from sklearn.cross_validation import train_test_split
from sklearn.utils import shuffle

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('epochs', 25, "The number of epochs.")
flags.DEFINE_integer('batch_size', 120, "The batch size.")
flags.DEFINE_string('road', 'hard', "Specify the simulator road")

TRAINING_FILE_CONFIG = {
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

MODEL_FILE = 'model.h5'
INPUT_SHAPE = (160,320,3)
STEERING_ADJUSTMENT = 0.25

# Load input image file names and training targets
def load_input():
  file_configs = TRAINING_FILE_CONFIG[FLAGS.road]
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
  return records

def image_augmentation(img_file_name, steering):
  X = []
  y = []
  img = mpimg.imread(img_file_name)
  X.append(img)
  y.append(steering)
  X.append(np.fliplr(img)) # flip the image for extra training data
  y.append(-steering)
  return X, y

# Generator for batch process
def batch_generator(input_records):
  X = []
  y = []
  while True:
    records = shuffle(input_records)
    for record in records:
      center_img_file_name = record['center']
      left_img_file_name   = record['left']
      right_img_file_name  = record['right']
      steering = record['steering']

      center_X, center_y = image_augmentation(center_img_file_name, steering)
      left_X, left_y = image_augmentation(left_img_file_name, steering + STEERING_ADJUSTMENT)
      right_X, right_y = image_augmentation(right_img_file_name, steering - STEERING_ADJUSTMENT)
      X += center_X + left_X + right_X
      y += center_y + left_y + right_y

      if len(X) >= FLAGS.batch_size:
        yield (np.array(X), np.array(y))
        X = []
        y = []


# def load_data():
#   file_configs = TRAINING_FILE_CONFIG[FLAGS.road]
#   records = []

#   for config in file_configs:
#     driving_log_file = config['log_file']
#     image_fold = config['image_fold']
#     reader = csv.reader(open(driving_log_file))
#     for row in reader:
#       r = {
#         'center' : image_fold + row[0].split('/')[-1],
#         'left'   : image_fold + row[1].split('/')[-1],
#         'right'  : image_fold + row[2].split('/')[-1],
#         'steering' : float(row[3]),
#         'throttle' : float(row[4]),
#         'brake'    : float(row[5]),
#         'speed'    : float(row[6])
#       }
#       records.append(r)

#   X = []
#   y = []
#   for record in records:
#     center_img_file_name = record['center']
#     left_img_file_name   = record['left']
#     right_img_file_name  = record['right']
#     steering = record['steering']
#     throttle = record['throttle']
#     brake    = record['brake']
#     speed    = record['speed']

#     img = mpimg.imread(center_img_file_name)
#     X.append(img)
#     y.append(steering)
#     X.append(np.fliplr(img)) # flip the image for extra training data
#     y.append(-steering)

#     img = mpimg.imread(left_img_file_name)
#     left_steering = steering + 0.25
#     X.append(img)
#     y.append(left_steering)
#     X.append(np.fliplr(img)) # flip the image for extra training data
#     y.append(-left_steering)

#     img = mpimg.imread(right_img_file_name)
#     right_steering = steering - 0.25
#     X.append(img)
#     y.append(right_steering)
#     X.append(np.fliplr(img)) # flip the image for extra training data
#     y.append(-right_steering)
#   return np.array(X), np.array(y)


# Build model based on nvidia's architecture from its white paper
def build_model():
  model = Sequential()
  model.add(Lambda(lambda x: (x / 128.0) - 1.0, input_shape=INPUT_SHAPE))
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
  return model


def main():
  model = build_model()
  checkpoint = ModelCheckpoint('model-{epoch:03d}.h5', monitor='val_loss', verbose=0, save_best_only=True, mode='auto')
  model.compile(loss='mean_squared_error', optimizer='adam')

  #X_train, y_train = load_data()

  # model.fit(X_train, y_train,
  #           epochs=FLAGS.epochs,
  #           batch_size=FLAGS.batch_size,
  #           validation_split=0.2,
  #           shuffle=True,
  #           callbacks=[checkpoint])

  train_data, valid_data = train_test_split(load_input(), test_size=0.2)

  # Times 6 because each record will produce 6 images/target pairs
  steps_per_epoch = len(train_data) * 6 / FLAGS.batch_size
  validation_steps = len(valid_data) * 6 / FLAGS.batch_size

  model.fit_generator(batch_generator(train_data),
                      steps_per_epoch,
                      validation_data = batch_generator(valid_data),
                      validation_steps = validation_steps,
                      callbacks=[checkpoint])
  model.save(MODEL_FILE)


if __name__ == '__main__':
  main()
