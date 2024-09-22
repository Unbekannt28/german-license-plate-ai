import tensorflow as tf
import cv2
import numpy as np
from image_utils import rgb_to_gray

TRAINING_IMAGES_PATH = "/home/unbekannt/Downloads/"
TEST_FILE = "image2.jpg"

model = tf.keras.models.load_model("german_license_plate_image_segmentation")

image = cv2.imread(TRAINING_IMAGES_PATH + TEST_FILE)
image = cv2.resize(image, (128, 128))
image = rgb_to_gray(image)
x_test = np.zeros((1, 128, 128, 1), dtype=np.uint8)
x_test[0] = image

cv2.imshow("Input", cv2.resize(image, (800, 800)))

#x_test = tf.keras.utils.normalize([image], axis=1)


preds_test = model.predict(x_test, verbose=1)

print(preds_test[0])

cv2.imshow("Output", cv2.resize(preds_test[0], (800, 800)))

overlay_image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
print(overlay_image)
for y, line in enumerate(preds_test[0]):
    for x, pixel in enumerate(line):
        if preds_test[0][y][x][0] > 0.4:
            overlay_image[y][x] = [255, 0, 0]
cv2.imshow("Overlayed", cv2.resize(overlay_image, (800, 800)))

cv2.waitKey(0)
cv2.destroyAllWindows()