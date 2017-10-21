import csv
import matplotlib.image as mpimg
#from sklearn.model_selection import train_test_split
import numpy as np

driving_log_file = 'data/driving_log.csv'

def load_data():
  reader = csv.DictReader(open(driving_log_file))
  X = []
  y = []
  for row in reader:
    img_file_name = 'data/' + row['center'].strip()
    img = mpimg.imread(img_file_name)
    X.append(img)
    y.append(float(row['steering'].strip()))
  return np.array(X), np.array(y)
