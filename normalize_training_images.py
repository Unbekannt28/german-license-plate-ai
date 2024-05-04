import cv2
import os

TRAINING_IMAGES_PATH = "training_images/"

for f in os.listdir(TRAINING_IMAGES_PATH):
    image = cv2.imread(TRAINING_IMAGES_PATH + f)
    height, width, _ = image.shape
    if height < width / 4 * 3:
        new_width = height / 3 * 4
        crop_x = ( width - new_width ) / 2
        image = image[:, int(crop_x):int(crop_x + new_width)]
    else:
        new_height = width / 4 * 3
        crop_y = ( height - new_height ) / 2
        image = image[int(crop_y):int(crop_y + new_height), :]
    image = cv2.resize(image, (640, 480))
    # cv2.imshow("Test", image)
    # break
    cv2.imwrite(TRAINING_IMAGES_PATH + f, image)

cv2.waitKey(0)
cv2.destroyAllWindows()