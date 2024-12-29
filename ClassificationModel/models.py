import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


def model_1(img_height, img_width, num_classes):

    # Adding random data augmentation of existing training data to prevent overfitting
    data_augmentation = keras.Sequential(
        [
            layers.Rescaling(1./255),
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ]
    )

    # Conv2D is a convolution function that extracts spatial features from the image input. Filter sets the amount of features/channels to output.
    # MaxPooling downsizes the amount of parameters by taking the max value of the window defined by pool size
    # Dropout randomly sets input to 0 to prevent overfitting the model. Only applies when training is set to True in call().
    # Flatten converts the input to one dimension which is required input for fully connected layers like Dense.

    model = Sequential([
        keras.Input(shape=(img_height, img_width, 3)),
        data_augmentation,
        layers.Conv2D(filters=16, kernel_size=(3,3), padding='same', activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(filters=32, kernel_size=(3,3), padding='same', activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(filters=16, kernel_size=(3,3), padding='same', activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.1),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dense(num_classes)
    ])

    return model



def model_2(img_height, img_width, num_classes):

    # Adding random data augmentation of existing training data to prevent overfitting
    data_augmentation = keras.Sequential(
        [
            layers.Rescaling(1./255),
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ]
    )

    model = keras.Sequential()
    model.add(keras.Input(shape=(img_height, img_width, 3))) 
    model.add(data_augmentation),
    model.add(layers.Conv2D(32, 5, strides=2, activation="relu"))
    model.add(layers.Conv2D(32, 3, activation="relu"))
    model.add(layers.MaxPooling2D(3))

    # Can you guess what the current output shape is at this point? Probably not.
    # Let's just print it:
    model.summary()

    # The answer was: (40, 40, 32), so we can keep downsampling...

    model.add(layers.Conv2D(32, 3, activation="relu"))
    model.add(layers.Conv2D(32, 3, activation="relu"))
    model.add(layers.MaxPooling2D(3))
    model.add(layers.Conv2D(32, 3, activation="relu"))
    model.add(layers.Conv2D(32, 3, activation="relu"))
    model.add(layers.MaxPooling2D(2))

    # And now?
    model.summary()

    # Now that we have 4x4 feature maps, time to apply global max pooling.
    model.add(layers.GlobalMaxPooling2D())

    # Finally, we add a classification layer.
    model.add(layers.Dense(num_classes))

    return model
