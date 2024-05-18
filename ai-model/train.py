import tensorflow as tf
import cv2
import os

TRAINING_IMAGES_PATH = "training_images_grayscale/"
LABEL_IMAGES_PATH = "label_images/"

print(tf.__version__)

x_train = []
for f in os.listdir(TRAINING_IMAGES_PATH):
    image = cv2.imread(TRAINING_IMAGES_PATH + f)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (320, 240))
    # print(image)
    # exit()
    x_train.append(image)

x_train = tf.keras.utils.normalize(x_train, axis=1)

y_train = []
for f in os.listdir(LABEL_IMAGES_PATH):
    image = cv2.imread(LABEL_IMAGES_PATH + f)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (320, 240))
    # print(image)
    # exit()
    y_train.append(image)

y_train = tf.keras.utils.normalize(y_train, axis=1)

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Flatten(input_shape=[240, 320]))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu)) # 128 neurons in the layer
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu)) # 128 neurons in the layer
model.add(tf.keras.layers.Dense(76800, activation=tf.nn.softmax)) # 10 neurons in the output layer (10 different possible results)

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.fit((x_train), (y_train), epochs=3)

model.save("numbers_example")

print("Model trained and saved.\nNow you can test it with tensorflow_predict.py!")