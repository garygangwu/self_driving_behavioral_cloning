import csv
import matplotlib.image as mpimg
#from sklearn.model_selection import train_test_split
import numpy as np

driving_log_file = 'data/driving_log.csv'
image_fold = 'data/IMG/'

def load_data():
  reader = csv.reader(open(driving_log_file))
  X = []
  y = []
  for row in reader:
    center_img_file_name = image_fold + row[0].split('/')[-1]
    left_img_file_name   = image_fold + row[1].split('/')[-1]
    right_img_file_name  = image_fold + row[2].split('/')[-1]
    steering = float(row[3])
    throttle = float(row[4])
    brake    = float(row[5])
    speed    = float(row[6])

    img = mpimg.imread(center_img_file_name)
    X.append(img)
    y.append(steering)

    # flip the image for extra training data
    img_flipped = np.fliplr(img)
    steering_flipped = -steering
    X.append(img_flipped)
    y.append(steering_flipped)

  return np.array(X), np.array(y)
