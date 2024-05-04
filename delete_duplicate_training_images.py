import os

TRAINING_IMAGES_PATH = "training_images/"
NEW_TRAINING_IMAGES_PATH = "images/gebrauchtwagen/"

known_filenames = []

for f in os.listdir(TRAINING_IMAGES_PATH):
    filename = ""
    
    for filename_part in f.split(".")[1:]:
        filename += filename_part
    
    known_filenames.append(filename)

for f in os.listdir(NEW_TRAINING_IMAGES_PATH):
    filename = ""
    
    for filename_part in f.split(".")[1:]:
        filename += filename_part

    if filename in known_filenames:
        os.remove(NEW_TRAINING_IMAGES_PATH + f)