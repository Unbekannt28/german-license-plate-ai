import cv2
import os
import numpy as np

TRAINING_IMAGES_PATH = "training_images/"
LABEL_IMAGES_PATH = "label_images/"

old_x = 0
old_y = 0
mouse_down = False
image = None
canvas = None

def draw(event, x, y, flags, parameters):
    global old_x, old_y, mouse_down, canvas, image

    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_down = True
    
    elif event == cv2.EVENT_LBUTTONUP:
        mouse_down = False
    
    elif event == cv2.EVENT_MOUSEMOVE and mouse_down:
        cv2.line(canvas, (old_x, old_y), (x, y), 255, thickness=8)
        cv2.line(image, (old_x, old_y), (x, y), 255, thickness=8)
    
    old_x = x
    old_y = y

window_name = "Label Imag"

for f in os.listdir(TRAINING_IMAGES_PATH):
    
    if f in os.listdir(LABEL_IMAGES_PATH):
        continue

    old_x = 0
    old_y = 0
    mouse_down = False
    image = cv2.imread(TRAINING_IMAGES_PATH + f)
    canvas = np.zeros((480, 640, 1), np.uint8)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, image)
    #cv2.resizeWindow(window_name, 600, 600)
    cv2.setMouseCallback(window_name, draw)

    while True:
        cv2.imshow(window_name, image)
        key = cv2.waitKey(1)
        if key == 27 or key == 13:
            break  # esc or enter to predict drawn number
    cv2.destroyAllWindows()

    cv2.imwrite(LABEL_IMAGES_PATH + f, canvas)