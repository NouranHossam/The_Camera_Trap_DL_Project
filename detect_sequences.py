import os
import glob
import shutil
import cv2 as cv
import numpy as np
import json
import math
from PIL import Image
from datetime import datetime, timedelta

# CAMERA TRAP CHALLENGE WS2022/23 - STUDY PROJECT:
# This script will process images find coherent image sequences in a given folder.

# Load all parameters from config file.
with open("config.json", "r") as jsonfile:
    config = json.load(jsonfile)

############################## TASK 1 - CREATE BDD DATASET ##############################

# Folder settings
destination_dir = config["destination_directory"]
background_dir = "backgrounds"
source_dir = ["dama_dama_damhirsch_deer/dama_dama_damhirsch", "meles_meles_dachs_badger/meles_meles_dachs"]

# Create directories.
if not os.path.exists(destination_dir):
    os.makedirs(destination_dir)
if not os.path.exists(background_dir):
    os.makedirs(background_dir)

# Copy animal images and empty backgrounds into separate folder. Print counts to verify.
for i in range(len(source_dir)):
    print("Source folder " + source_dir[i] + ": " + str(len(os.listdir(source_dir[i] + "/dayvision"))) + " images.\n" +
          "Empty folder with backgrounds: " + str(len(os.listdir(source_dir[i] + "/empty/day"))) + " images.")
    for image in glob.iglob(os.path.join(source_dir[i] + "/dayvision", "*.jpg")):
        if image not in glob.iglob(os.path.join(destination_dir, "*.jpg")):
            shutil.copy(image, destination_dir)
    for image in glob.iglob(os.path.join(source_dir[i] + "/empty/day", "*.jpg")):
        if image not in glob.iglob(os.path.join(background_dir, "*.jpg")):
            shutil.copy(image, background_dir)

print(destination_dir + " folder with animals combined: " + str(len(os.listdir(destination_dir))) + " images.\n" +
      "Background folder with empty images combined: " + str(len(os.listdir(background_dir))) + " images.")


############################## TASK 2 - IDENTIFY IMAGE SEQUENCES ##############################

#From here on, work with new BDD folder as dataset.
image_filenames = os.listdir(destination_dir)

# Create list that keeps track of images that are checked (outer loop) or found as similar (inner loop).
# Using Camera name, image datetime (and possibly also RMSE) for sequence detection. Save them 2D array.
images_to_check = image_filenames.copy()
all_sequences = []

# Iterate through all images.
for i in range(len(image_filenames)):

    # Only proceed if sequence of image still needs to be identified.
    if (image_filenames[i] in images_to_check):

        # Current image is the image that we check for similarity.
        current_image_path = os.path.join(destination_dir, image_filenames[i])
        current_image = cv.imread(current_image_path, cv.IMREAD_GRAYSCALE)
        # Extract datetime information and turn into datetime object.
        current_image_exif = Image.open(current_image_path)
        exif = current_image_exif.getexif()
        current_date_time = exif.get(36867)
        current_date_time = datetime.strptime(current_date_time, "%Y:%m:%d %H:%M:%S")

        # Extract camera name in bottom left corner.
        height, width = current_image.shape
        current_camera_name = current_image[int(height * 0.98):, :int(width * 0.2)]

        # Update status.
        images_to_check[i] = "checked"

        # Create array that collects images of the same sequence.
        same_sequence_images = [(image_filenames[i])]

        # Iterate through all images that do not have a sequence yet and load as other image.
        for image in (os.listdir(destination_dir)):
            if (image in images_to_check):
                other_image_path = os.path.join(destination_dir, image)
                other_image = cv.imread(other_image_path, cv.IMREAD_GRAYSCALE)
                # Extract datetime information and turn into datetime object.
                other_image_exif = Image.open(other_image_path)
                exif = other_image_exif.getexif()
                other_date_time = exif.get(36867)
                other_date_time = datetime.strptime(other_date_time, "%Y:%m:%d %H:%M:%S")

                # Extract camera name in bottom left corner.
                other_camera_name = other_image[int(height * 0.98):, :int(width * 0.2)]
                camera_name_difference = other_camera_name.copy()

                # Check if there is a difference in the camera name.
                cv.absdiff(current_camera_name, other_camera_name, camera_name_difference)
                white_pixels = np.sum(camera_name_difference == 255)

                # Calculate RMSE (currently not used for sequence identification).
                #MSE = np.square(np.subtract(current_image, other_image)).mean()
                #RMSE = math.sqrt(MSE)
                #print("Similarity of " + str(RMSE) + " between images " + image + " & " + image_filenames[i])

                # Time tolerance can be adjusted for datetime similarity.
                before_tolerance = current_date_time - timedelta(minutes = config["time_tolerance"])
                after_tolerance = current_date_time + timedelta(minutes = config["time_tolerance"])

                # Here we can switch between using RMSE+datetime or Camera-name+datetime for sequence identification.
                #if (RMSE < config["rmse_threshold_with_animal"] and other_date_time < after_tolerance and other_date_time > before_tolerance):
                if (white_pixels < 10 and other_date_time < after_tolerance and other_date_time > before_tolerance):
                    if not (image == image_filenames[i]):
                        same_sequence_images.append(image)
                        images_to_check[os.listdir(destination_dir).index(image)] = "found"

        # For each identified sequence, print image names and sequence size.
        print(str(len(same_sequence_images)) + " images of same sequence found: " + str(same_sequence_images))
        all_sequences.append(same_sequence_images)

# Store the identified sequences in a text file.
with open("all_sequences.txt", "w") as file:
    for line in all_sequences:
        file.write(" ".join(line) + "\n")