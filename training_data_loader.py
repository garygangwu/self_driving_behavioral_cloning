import csv
import matplotlib.image as mpimg
#from sklearn.model_selection import train_test_split
import numpy as np

#driving_log_file = 'data/driving_log.csv'
#image_fold = 'data/IMG/'
driving_log_file = 'data_hard_additional/driving_log.csv'
image_fold = 'data_hard_additional/IMG/'
driving_log_file_1 = 'data_tough_roads/driving_log.csv'
image_fold_1 = 'data_tough_roads/IMG/'
driving_log_file_2 = 'data_hard_additional_2/driving_log.csv'
image_fold_2 = 'data_hard_additional_2/IMG/'

def load_data():
  reader = csv.reader(open(driving_log_file))
  records = []
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

  reader = csv.reader(open(driving_log_file_1))
  for row in reader:
    r = {
      'center' : image_fold_1 + row[0].split('/')[-1],
      'left'   : image_fold_1 + row[1].split('/')[-1],
      'right'  : image_fold_1 + row[2].split('/')[-1],
      'steering' : float(row[3]),
      'throttle' : float(row[4]),
      'brake'    : float(row[5]),
      'speed'    : float(row[6])
    }
    records.append(r)

  reader = csv.reader(open(driving_log_file_2))
  for row in reader:
    r = {
      'center' : image_fold_2 + row[0].split('/')[-1],
      'left'   : image_fold_2 + row[1].split('/')[-1],
      'right'  : image_fold_2 + row[2].split('/')[-1],
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
