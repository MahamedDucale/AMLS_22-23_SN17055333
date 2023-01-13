import tensorflow as tf
import os
import pandas as pd

# Define the directory where the training images are stored
train_data_dir = "datasets/cartoon_set"
# Define the directory where the testing images are stored
test_data_dir = "datasets/cartoon_set_test"

# Create a list of file paths to the training images
train_image_paths = [os.path.join(train_data_dir, file_name) for file_name in os.listdir(train_data_dir) if file_name.endswith(".png")]

# Create a list of file paths to the testing images
test_image_paths = [os.path.join(test_data_dir, file_name) for file_name in os.listdir(test_data_dir) if file_name.endswith(".png")]

# Create a dataset from the file paths
train_dataset = tf.data.Dataset.from_tensor_slices(train_image_paths)
test_dataset = tf.data.Dataset.from_tensor_slices(test_image_paths)

# Use the dataset to read and decode the images
def read_and_decode(file_path):
    image_string = tf.io.read_file(file_path)
    image_tensor = tf.image.decode_png(image_string, channels=3)
    return image_tensor

train_dataset = train_dataset.map(read_and_decode)
test_dataset = test_dataset.map(read_and_decode)

# Load the labels and exclude the first row
train_labels = pd.read_csv("datasets/cartoon_set/labels.csv", skiprows=[0])
test_labels = pd.read_csv("datasets/cartoon_set_test/labels.csv", skiprows=[0])

# Extract the gender labels
print(train_labels[0])