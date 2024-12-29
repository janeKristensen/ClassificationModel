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
import models

from tensorflow.python import training
from tensorflow.python.keras.layers.core import regularizers


def show_results(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(len(history.epoch))

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
        image = image.resize((256,256))
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


# Create training dataset from directories
def get_data(path, batch_size):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        path,
        validation_split=0.2,
        subset="training",
        seed=123,
        shuffle=True,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    val_ds = tf.keras.utils.image_dataset_from_directory(
        path,
        validation_split=0.2,
        subset="validation",
        seed=123,
        shuffle=True,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    class_names = train_ds.class_names

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # print(len(train_ds))
    # for images, labels in train_ds.take(1):
    #     plt.imshow(images[0].numpy().astype(np.uint8))
    #     plt.show()
    #     print(labels.numpy())

    return train_ds, val_ds, class_names



# Compile and fit the model 
def build_model(path, img_height, img_width, epochs, batch_size):
   
    train_ds, val_ds, class_names = get_data(data_dir, batch_size)
    num_classes = len(class_names)

    # Get the sequential model to use
    model = models.model_1(img_height, img_width, num_classes)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001, clipvalue=1.0), 
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'],
    )

    model.summary()

    # Fit the model on training data
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    class_weights = {0: 1.0, 1: 1.5, 2: 1.2}

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[early_stopping],
        class_weight=class_weights,
    )

    show_results(history)

    return model, class_names



# Load test data and predict labels
batch_size = 32
img_height = 256
img_width = 256
epochs = 50

data_dir = "D:\Visual Studio stuff\Projekts\MachineLearning\ClassificationModel\Images"
model, class_names = build_model(data_dir, img_height, img_width, epochs, batch_size);


test_data_path = "D:\\Visual Studio stuff\\Projekts\\MachineLearning\\ClassificationModel\\ClassificationModel\\Unknown"
scale_image(test_data_path)
test_data = [file for file in os.listdir(test_data_path) if os.path.isfile(os.path.join(test_data_path, file))]

for file in test_data:
    image_data = load_image(os.path.join(test_data_path, file), img_height, img_width)
    predictions = model.predict(image_data)
    print(predictions)
    score = tf.nn.softmax(predictions[0])
  
    print("Predicted scores: Cat {:.4f}, Dog {:.4f}, My Cat {:.4f}"
          .format(score[0], score[1], score[2]))
    print("The image {} most likely belongs to {} with a {:.2f} percent confidence." 
          .format(file, class_names[np.argmax(score)], 100 * np.max(score))
    )


 
