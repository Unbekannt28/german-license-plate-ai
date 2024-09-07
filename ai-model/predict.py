import tensorflow as tf
import cv2
import numpy as np

TRAINING_IMAGES_PATH = "../training_images_grayscale/"
TEST_FILE = "1.VW-Touran-1200x800-25c20617a810e9f5.jpg"

model = tf.keras.models.load_model("german_license_plate_image_segmentation")

image = cv2.imread(TRAINING_IMAGES_PATH + TEST_FILE)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

x_test = tf.keras.utils.normalize([image], axis=1)

preds_test = model.predict(x_test, verbose=1)

cv2.imshow("Output", preds_test[0])
cv2.waitKey(0)
cv2.destroyAllWindows()