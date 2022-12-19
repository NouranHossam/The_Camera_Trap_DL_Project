import os
import glob
import shutil
import cv2 as cv
import numpy as np
import math
import imutils
import json
from PIL import Image
from datetime import datetime, timedelta

# CAMERA TRAP CHALLENGE WS2022/23 - STUDY PROJECT:
# The script will process images in the following order: finding coherent sequences,
# locating animals in in the images, classifying animals into different categories.

# Load all parameters from config file.
with open("config.json", "r") as jsonfile:
    config = json.load(jsonfile)

# TASK 1 - CREATE BDD DATASET

if not os.path.exists("BDD"):
    os.makedirs("BDD")

source_dir = ["dama_dama_damhirsch_deer/dama_dama_damhirsch/dayvision", "meles_meles_dachs_badger/meles_meles_dachs/dayvision"]
destination_dir = "BDD"

for i in range(len(source_dir)):
    print("Source folder " + str(i+1) + ": " + str(len(os.listdir(source_dir[i]))) + " images.")
    for image in glob.iglob(os.path.join(source_dir[i], "*.jpg")):
        if image not in glob.iglob(os.path.join(destination_dir, "*.jpg")):
            shutil.copy(image, destination_dir)

print("Destination folder: " + str(len(os.listdir(destination_dir))) + " images.")

#From here on, work with new folder as dataset
image_filenames = os.listdir(destination_dir)

# TASK 2 - IDENTIFY COHERENT IMAGE SEQUENCES (comment task 2 for runtime if working on other tasks!!!)

# Create list that keeps track of images that are checked (outer loop) or found as similar (inner loop).
# Save the sequences in 2D array.
images_to_check = image_filenames.copy()
all_sequences = []

# Iterate through all images.
for i in range(len(image_filenames)):

    # Only proceed if sequence of image still needs to be identified.
    if (image_filenames[i] in images_to_check):

        # Current image is the image that we check for similarity.
        current_image_path = os.path.join(destination_dir, image_filenames[i])
        current_image = cv.imread(current_image_path, cv.IMREAD_GRAYSCALE)
        current_image_exif = Image.open(current_image_path)
        exif = current_image_exif.getexif()
        current_date_time = exif.get(36867)
        current_date_time = datetime.strptime(current_date_time, "%Y:%m:%d %H:%M:%S")

        # Update status.
        images_to_check[i] = "checked"

        # Create array that collects images of the same sequence.
        same_sequence_images = [(image_filenames[i])]

        # Iterate through all images that do not have a sequence yet.
        for image in (os.listdir(destination_dir)):
            # Use OpenCV for image processing and Image for exif datetime.
            if (image in images_to_check):
                other_image_path = os.path.join(destination_dir, image)
                other_image = cv.imread(other_image_path, cv.IMREAD_GRAYSCALE)
                other_image_exif = Image.open(other_image_path)
                exif = other_image_exif.getexif()
                other_date_time = exif.get(36867)
                other_date_time = datetime.strptime(other_date_time, "%Y:%m:%d %H:%M:%S")

                # Calculate RMSE.
                MSE = np.square(np.subtract(current_image, other_image)).mean()
                RMSE = math.sqrt(MSE)

                # Before & after tolerance can be adjusted for datetime similarity.
                before_tolerance = current_date_time - timedelta(hours = config["time_tolerance"])
                after_tolerance = current_date_time + timedelta(hours = config["time_tolerance"])

                # RMSE threshold can be adjusted here for more/less image similarity.
                if (RMSE < config["rmse_threshold"] and other_date_time < after_tolerance and other_date_time > before_tolerance):
                    if not (image == image_filenames[i]):
                        same_sequence_images.append(image)
                        images_to_check[os.listdir(destination_dir).index(image)] = "found"

        # For each identified sequence, print image names and sequence size.
        print(str(len(same_sequence_images)) + " images of same sequence found: " + str(same_sequence_images))
        all_sequences.append(same_sequence_images)

# TASK 3 - LOCATING ANIMAL IN IMAGE

# Testing with first detected sequence from task 2. Using predefined array to save runtime.
sequence_1 = ['IMG2_0647.jpg', 'IMG_0136.JPG', 'IMG_0137.JPG', 'IMG_0138.JPG', 'IMG_0139.JPG', 'IMG_0140.JPG', 'IMG_0141.JPG', 'IMG_0142.JPG', 'IMG_0143.JPG', 'IMG_0144.JPG', 'IMG_0145.JPG', 'IMG_0146.JPG', 'IMG_0147.JPG', 'IMG_0148.JPG', 'IMG_0149.JPG', 'IMG_0150.JPG']
sequence_1_images = []
sequence_1_datetime = []

# Create array with all images in the sequence.
for image in sequence_1:
    # To save image in sequence array.
    current_image_path = os.path.join(destination_dir, image)
    current_image = cv.imread(current_image_path, cv.IMREAD_GRAYSCALE)
    # This is just to cut out top 25% of the image for better results. Delete of not wanted.
    # TODO - Figure out how to correctly calculate back to original image frame.
    height, width = current_image.shape
    #print(str(height) + " x " + str(width))
    current_image = current_image[int(height*config["image_cutting"]): height, 0: width]
    sequence_1_images.append(current_image)

# Calculate average image from all images.
sequence_1_average = sequence_1_images[0]

for image in range(len(sequence_1_images)):
    # First image already included above.
    if image == 0:
        pass
    else:
        # Check again how this exactly works - three lines copied from Stack Overflow.
        alpha = 1.0 / (image + 1)
        beta = 1.0 - alpha
        sequence_1_average = cv.addWeighted(sequence_1_images[image], alpha, sequence_1_average, beta, 0.0)

# Iterating over all images in sequence again to compare it with sequence average.
for i in range(10, len(sequence_1_images)):
    interest_image = sequence_1_images[i]

    # Get absolute difference between image and average.
    difference_image = sequence_1_average.copy()
    cv.absdiff(interest_image, sequence_1_average, difference_image)
    cv.imshow("difference", difference_image)
    cv.waitKey(0)
    cv.destroyAllWindows()

    # To help identify animal location, defining thresholds for binarization.
    (T, binary_image) = cv.threshold(difference_image, config["binarization_lower"], config["binarization_upper"], cv.THRESH_BINARY)
    cv.imshow("binary", binary_image)
    cv.waitKey(0)
    cv.destroyAllWindows()

    # Erode white pixels to get rid of some background noise. Here, in binarization and erosion
    # of the image, some parameters need to be adjusted and tested for different thresholds.
    kernel = np.ones((config["erosion_kernel"], config["erosion_kernel"]), np.uint8)
    erosion_image = cv.erode(binary_image, kernel, iterations = config["erosion_iterations"])
    cv.imshow("erosion", erosion_image)
    cv.waitKey(0)
    cv.destroyAllWindows()

    # Finding the contours that are left after processing, hopefully only animal.
    contours = cv.findContours(erosion_image, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    # Arrays to store x/y of all contours to calculate mean centroid.
    centroid_x = []
    centroid_y = []

    # Iterating through all contours and drawing circles.
    for c in contours:
        (x, y), radius = cv.minEnclosingCircle(c)
        center = (int(x), int(y))
        radius = int(radius*2)
        # To get rid of contours in top black panel. Need to readjust.
        if (y > 100):
            centroid_x.append(int(x))
            centroid_y.append(int(y))
        cv.circle(interest_image, center, radius, (0, 255, 0), 10)

    #print(centroid_x)

    # Calculate mean centroid only if there is pixels left and draw as big circle.
    if centroid_x:
        mean_centroid_x = int(np.mean(centroid_x))
        mean_centroid_y = int(np.mean(centroid_y))
        mean_centroid = (mean_centroid_x, mean_centroid_y)
        cv.circle(interest_image, mean_centroid, 50, (0, 255, 0), 10)

    # Display each image with the calculated centroids to see results.
    interest_image = cv.resize(interest_image, (1200, 700))
    cv.imshow("Sequence 1 / Image " + str(i+1) + " - Remaining contour centroids after binarization and erosion", interest_image)
    cv.waitKey(0)
    cv.destroyAllWindows()

''' 
TODOs FROM HERE ON: 
- Check if code for task 3 also works for other image sequences
- Test different parameters in task 3 and also threshold in task 2 for correct sequences
- Make use of datetime objects above to validate image order
- Task 4
'''
