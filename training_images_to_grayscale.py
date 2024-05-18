import cv2
import os

TRAINING_IMAGES_PATH = "training_images/"
TRAINING_IMAGES_GRAYSCALE_PATH = "training_images_grayscale/"

for f in os.listdir(TRAINING_IMAGES_PATH):
    image = cv2.imread(TRAINING_IMAGES_PATH + f)
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # cv2.imshow("Test", image)
    # break
    cv2.imwrite(TRAINING_IMAGES_GRAYSCALE_PATH + f, image)

cv2.waitKey(0)
cv2.destroyAllWindows()