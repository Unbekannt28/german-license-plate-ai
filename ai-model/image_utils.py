import numpy as np

def rgb_to_gray(image):
    gray_image = np.zeros((len(image), len(image[0]), 1), dtype=np.uint8)
    for y, line in enumerate(image):
        for x, pixel in enumerate(line):
            gray_image[y][x] = [pixel[0]]
    return gray_image