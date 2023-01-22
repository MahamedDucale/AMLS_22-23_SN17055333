import os
import random
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import pandas as pd 
from keras.applications.efficientnet import preprocess_input
from keras.utils import to_categorical

def preprocess(path, label):
    label = tf.strings.to_number(label, out_type=tf.int32)
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [224, 224])
    # Data augmentation
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    image = tf.image.random_hue(image, max_delta=0.1)
    image = tf.image.random_saturation(image, lower=0.9, upper=1.1)
    image = tf.clip_by_value(image, 0, 1)
    return image, label



def plot_training_history(history):
    # Plot the training and validation accuracy over the epochs
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig('B1_train.png')

    # Plot the training and validation loss over the epochs
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig('B1_val.png')

def create_data(train_img_dir, test_img_dir, index, train_labels_path, test_labels_path, val_percentage):
    # Load the labels
    train_labels = pd.read_csv(train_labels_path).replace('\t',' ', regex=True).values.tolist()
    test_labels = pd.read_csv(test_labels_path).replace('\t',' ', regex=True).values.tolist()
    train_labels = [int(sublist[0].split()[index]) for sublist in train_labels]
    test_labels = [int(sublist[0].split()[index]) for sublist in test_labels]
    train_img_filenames = os.listdir(train_img_dir)
    train_img_filenames = sorted(train_img_filenames, key=lambda x: int(x.split(".")[0]))
    test_img_filenames = os.listdir(test_img_dir)
    test_img_filenames = sorted(test_img_filenames, key=lambda x: int(x.split(".")[0]))

    train_data = list(map(list, zip(train_img_filenames, map(str,train_labels))))
    random.shuffle(train_data)
    test_data = list(map(list, zip(test_img_filenames, map(str,test_labels))))
    random.shuffle(test_data)
    # Calculate the number of validation labels
    val_count = int(len(train_labels) * val_percentage)
    # Split the list of labels into train and validation sets
    val_data = train_data[:val_count]
    train_data = train_data[val_count:]

    return train_data, val_data, test_data

train_img_dir = "Datasets/cartoon_set/img/"
test_img_dir = "Datasets/cartoon_set_test/img/"
index = 2
train_labels_path = "Datasets/cartoon_set/labels.csv"
test_labels_path = "Datasets/cartoon_set_test/labels.csv"
val_percentage = 0.2
train_data,val_data,test_data = create_data(train_img_dir,test_img_dir,index,train_labels_path,test_labels_path,val_percentage)
#create datasets
train_dataset = tf.data.Dataset.from_tensor_slices(train_data)
train_dataset = train_dataset.map(lambda x: (tf.strings.join([train_img_dir, x[0]]), x[1]))
val_dataset = tf.data.Dataset.from_tensor_slices(val_data)
val_dataset = val_dataset.map(lambda x: (tf.strings.join([train_img_dir, x[0]]), x[1]))

test_dataset = tf.data.Dataset.from_tensor_slices(test_data)
test_dataset = test_dataset.map(lambda x: (tf.strings.join([test_img_dir, x[0]]), x[1]))

train_dataset = train_dataset.map(preprocess)
val_dataset = val_dataset.map(preprocess)
test_dataset = test_dataset.map(preprocess)

#create data generator
batch_size = 32

train_dataset = train_dataset.shuffle(buffer_size=len(train_data))
train_dataset = train_dataset.batch(batch_size)
val_dataset = val_dataset.batch(batch_size)
test_dataset = test_dataset.batch(batch_size)

# Load a pre-trained model
base_model = tf.keras.applications.EfficientNetB0(weights='imagenet', include_top=False, input_shape=[224, 224, 3])

# Freeze the base_model
base_model.trainable = False

# Create new model on top
inputs = keras.Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = keras.layers.GlobalAveragePooling2D()(x)
outputs = keras.layers.Dense(5, activation="softmax")(x)
model = keras.Model(inputs, outputs)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


model.summary()

history = model.fit(train_dataset, epochs=10, validation_data=val_dataset)

plot_training_history(history)

test_loss, test_acc = model.evaluate(test_dataset)
print('Test accuracy:', test_acc)