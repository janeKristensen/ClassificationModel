from genericpath import isfile
from random import shuffle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import Sequential
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os


def show_results(history, epochs):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()


def scale_image(directory):
    for item in os.listdir(directory):
        path = directory + "\\" + item
        image = Image.open(path)
        image = image.resize((250,250))
        image.save(path)


def normalize(data):
    normalized_data = data/255
    print(normalized_data)
    return normalized_data


def plot_image(image):
    image = normalize(image)
    plt.imshow(image)
    plt.show()


def load_image(image_path, img_height, img_width):
    img = tf.keras.utils.load_img(
        image_path, target_size=(img_height, img_width)
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch
    return img_array


def get_data(path, batch_size):
    data = tf.keras.utils.image_dataset_from_directory(
        path,
        validation_split=0.2,
        subset="training",
        seed=123,
        shuffle=True,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    class_names = data.class_names
    
    AUTOTUNE = tf.data.AUTOTUNE
    train_size = int(len(data) * 0.7)
    val_size = int(len(data) * 0.3)

    train_ds = data.take(train_size).shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = data.skip(train_size).take(val_size).prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, class_names


def build_model(path, img_height, img_width, epochs, batch_size):
   
    train_ds, val_ds, class_names = get_data(data_dir, batch_size)
    num_classes = len(class_names)

    model = Sequential([
      layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
      layers.Conv2D(32, 3, padding='same', activation='relu'),
      layers.MaxPooling2D(),
      layers.Dropout(0.1),
      layers.Conv2D(64, 3, padding='same', activation='relu'),
      layers.MaxPooling2D(),
      layers.Flatten(),
      layers.Dense(128, activation='relu'),
      layers.Dense(num_classes)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.summary()

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )

    show_results(history, epochs)

    return model, class_names


batch_size = 32
img_height = 250
img_width = 250
epochs = 10

data_dir = "D:\Visual Studio stuff\Projekts\MachineLearning\ClassificationModel\Images"
model, class_names = build_model(data_dir, img_height, img_width, epochs, batch_size);


test_data_path = "D:\\Visual Studio stuff\\Projekts\\MachineLearning\\ClassificationModel\\ClassificationModel\\Unknown"
test_data = [file for file in os.listdir(test_data_path) if os.path.isfile(os.path.join(test_data_path, file))]

for file in test_data:
    image_data = load_image(os.path.join(test_data_path, file), img_height, img_width)
    predictions = model.predict(image_data)
    score = tf.nn.softmax(predictions[0])

    print(
        "This image {} most likely belongs to {} with a {:.2f} percent confidence."
        .format(file, class_names[np.argmax(score)], 100 * np.max(score))
    )
