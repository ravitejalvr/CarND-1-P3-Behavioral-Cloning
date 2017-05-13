import gc

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Conv2D, K
from keras.models import Sequential
from keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


def import_data(csv_path='./data/driving_log.csv'):
    return pd.read_csv(csv_path)


def img_change_brightness(img, brightness_range=0.1):
    temp = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Compute a random brightness value and apply it to the image
    brightness = brightness_range + np.random.uniform() - 0.5
    temp[:, :, 2] = temp[:, :, 2] * brightness

    return cv2.cvtColor(temp, cv2.COLOR_HSV2RGB)


def img_translate(img, x_translation, trans_y_range=20):
    # Randomly compute a Y translation
    y_translation = (trans_y_range * np.random.uniform()) - (trans_y_range / 2)

    # Form the translation matrix
    translation_matrix = np.float32([[1, 0, x_translation], [0, 1, y_translation]])

    # Translate the image
    return cv2.warpAffine(img, translation_matrix, (img.shape[1], img.shape[0]))


def data_augment(img_path, angle, trans_x_range=50, angel_per_trans=0.1):
    # Randomly form the X translation distance and compute the resulting steering angle change
    x_translation = (trans_x_range * np.random.uniform()) - (trans_x_range / 2)
    new_angle = angle + ((x_translation / trans_x_range) * 2) * angel_per_trans

    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    img = img_change_brightness(img)
    img = img_translate(img, x_translation)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)  # YUV as noted in nvidia's network architecture

    # flip the image (50/50 chance)
    if np.random.randint(2) == 0:
        return img, new_angle
    return np.fliplr(img), -new_angle


# generator feeds augmented training and validation data to the model
def generator(df, batch_size=32, data_path='./data/IMG/'):
    # create adjusted steering measurements for the side camera images
    correction = 0.25
    steering_params = [0, correction, -correction]

    header = list(df.columns.values)

    images = []
    angles = []
    while True:
        idx = np.random.randint(len(df))

        image_pos = np.random.randint(3)  # random center, left or right

        image = data_path + df[header[image_pos]].iloc[idx].split('\\')[-1]
        angle = df.steering.iloc[idx] + steering_params[image_pos]

        image, angle = data_augment(image, angle)
        images.append(image)
        angles.append(angle)

        if len(images) >= batch_size:
            yield shuffle(np.array(images), np.array(angles))
            images, angles = [], []


def model_nvidia():
    # Keras Model Inisilization
    model = Sequential()

    # Cropping image to get ride of important information (e.g trees, hills ,engine compartment)
    model.add(Cropping2D(cropping=((61, 23), (0, 0)), input_shape=(160, 320, 3)))  # 61, 23

    # normalize and mean center each pixel [-0.5, 0.5]
    model.add(Lambda(lambda x: (x / 255.0) - 0.5))

    model.add(Conv2D(24, kernel_size=(5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(36, kernel_size=(5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(48, kernel_size=(5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))

    model.add(Flatten())

    # model.add(Dense(1164, W_regularizer=l2(0.001)))
    model.add(Dense(100, W_regularizer=l2(0.001)))
    model.add(Dense(50, W_regularizer=l2(0.001)))
    model.add(Dense(10, W_regularizer=l2(0.001)))

    model.add(Dense(1, W_regularizer=l2(0.001)))

    # Compile model with mean squared error and adam optimizer
    model.compile(loss='mse', optimizer='adam')

    print(model.summary())
    return model


def train(data, model, filename='model.h5', batch_size=512, test_size=0.1, epochs=3):
    train_data, validation_data = train_test_split(data, test_size=test_size)

    # compile and train the model using the generator function
    train_generator = generator(train_data, batch_size)
    validation_generator = generator(validation_data, batch_size)

    history_object = model.fit_generator(train_generator, steps_per_epoch=len(train_data) // batch_size,
                                         validation_data=validation_generator,
                                         validation_steps=len(validation_data) // batch_size,
                                         epochs=epochs)

    with open('model.json', 'w') as f:
        f.write(model.to_json())

    model.save(filename)

    # plot the training and validation loss for each epoch
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()

    K.clear_session()
    gc.collect()


if __name__ == '__main__':
    data = import_data()
    train(data, model=model_nvidia(), filename='nvidia.h5', epochs=1)
