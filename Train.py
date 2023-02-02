import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from sklearn import preprocessing
import joblib
import cv2

# Define the path to the dataset
dataset_path = 'facedataset2'

# Create a list to store the images and labels
images = []
labels = []

# Iterate through the subdirectories in the dataset
for subdir in os.listdir(dataset_path):
    if subdir != 'train' and subdir != 'validation':
        subdir_path = os.path.join(dataset_path, subdir)

        for image_name in os.listdir(subdir_path):
            # Load the image and resize it to (128, 128)

            image_path = os.path.join(subdir_path, image_name)

            image = cv2.imread(image_path)
            image = cv2.resize(image, (128, 128))

            # Append the image and its label to the lists
            images.append(image)
            labels.append(subdir)

# Convert the lists to numpy arrays
images = np.array(images)
labels = np.array(labels)

# Encode the labels
le = preprocessing.LabelEncoder()
labels = le.fit_transform(labels)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.5, random_state=42)

