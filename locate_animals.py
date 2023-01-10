import os
import cv2 as cv
import numpy as np
import math
import imutils
import json
from matplotlib import pyplot as plt

# CAMERA TRAP CHALLENGE WS2022/23 - STUDY PROJECT:
# This script will locate animals in the previously found sequences.

# Load all parameters from config file.
with open("config.json", "r") as jsonfile:
    config = json.load(jsonfile)

# Folder settings.
destination_dir = config["destination_directory"]
background_dir = "backgrounds"

############################## TASK 3 - LOCATING ANIMAL IN IMAGE ##############################

# Load image sequences from text file to list again.
all_sequences = []

with open("all_sequences.txt") as file:
    for line in file:
        all_sequences.append(line.split())

#To test badger vs deer.
#all_sequences = [["IMG_2547.JPG", "IMG_2548.JPG", "IMG_2549.JPG", "IMG_2550.JPG", "IMG_2551.JPG"], ["IMG_2079.JPG", "IMG_2080.JPG", "IMG_2081.JPG", "IMG_2082.JPG", "IMG_2083.JPG", "IMG_2084.JPG"]]

# Loop through all sequences.
for sequence in all_sequences:

    print("Checking image sequence: " + str(sequence))
    sequence = sequence
    sequence_images = []
    sequence_images_color = []
    sequence_images_modified = []

    # Create array with all images in the sequence.
    for image in sequence:

        # To save image in sequence array.
        current_image_path = os.path.join(destination_dir, image)
        current_image = cv.imread(current_image_path, cv.IMREAD_GRAYSCALE)
        current_image_color = cv.imread(current_image_path, cv.IMREAD_COLOR)

        # Append to images list.
        sequence_images.append(current_image)
        sequence_images_color.append(current_image_color)

    # Only do calculations if a sequence could be detected.
    if (len(sequence) > 1):

        # Calculate average image from all images.
        sequence_average = sequence_images[0]

        for image in range(len(sequence_images)):

            # First image already included above.
            if image == 0:
                pass
            else:
                # Check again how this exactly works - three lines copied from Stack Overflow.
                alpha = 1.0 / (image + 1)
                beta = 1.0 - alpha
                sequence_average = cv.addWeighted(sequence_images[image], alpha, sequence_average, beta, 0.0)

        # Compare sequence average to empty backgrounds.
        background_images = os.listdir(background_dir)

        # Keep track of lowest RMSE.
        lowest_rmse_value = 10
        lowest_rmse_index = ""

        # Loop through all empty backgrounds and calculate lowest RMSE between background and average image.
        for background in background_images:

            background_path = os.path.join(background_dir, background)
            background_image = cv.imread(background_path, cv.IMREAD_GRAYSCALE)

            # Calculate RMSE for similarity.
            MSE = np.square(np.subtract(background_image, sequence_average)).mean()
            RMSE = math.sqrt(MSE)

            if (RMSE < lowest_rmse_value):
                lowest_rmse_value = RMSE
                lowest_rmse_index = background

        # Use background not average image if RMSE lower than x.
        if (lowest_rmse_value < config["rmse_threshold_empty_background"]):
            background_path = os.path.join(background_dir, lowest_rmse_index)
            background_image = cv.imread(background_path, cv.IMREAD_GRAYSCALE)
            sequence_average = background_image
            print("Using background image: " + lowest_rmse_index)
        else:
            print("Using average image")

        # Iterating over all images in sequence again to compare it with sequence average.
        for i in range(0, len(sequence_images)):

            interest_image = sequence_images[i]
            interest_image_color = sequence_images_color[i]

            # Get absolute difference between image and average.
            difference_image = sequence_average.copy()
            cv.absdiff(interest_image, sequence_average, difference_image)

            # Cut out top part of image for better results, size can be adjusted in config.
            height, width = difference_image.shape

            # To help identify animal location, defining thresholds for binarization.
            (T, binary_image) = cv.threshold(difference_image, config["binarization_lower"], config["binarization_upper"], cv.THRESH_BINARY)

            # Turn top part of binary image black to avoid tree noise. Threshold adjustable.
            binary_image[:int(height * config["image_cutting"]), :] = 0

            # Erode white pixels to get rid of some background noise. Here, in binarization and erosion
            # of the image, some parameters need to be adjusted and tested for different thresholds.
            kernel = np.ones((config["erosion_kernel"], config["erosion_kernel"]), np.uint8)
            erosion_image = cv.erode(binary_image, kernel, iterations = config["erosion_iterations"])

            dilation_image = cv.dilate(erosion_image, kernel, iterations = config["dilation_iterations"])

            # Finding the contours that are left after processing, hopefully only animal.
            contours = cv.findContours(dilation_image, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)   # can also take contours of eroded image.
            contours = imutils.grab_contours(contours)

            # If there are contours left after erosion, identify the biggest one and draw on image.
            if len(contours) != 0:
                biggest_contour = max(contours, key = cv.contourArea)
                x, y, w, h = cv.boundingRect(biggest_contour)
                cv.rectangle(interest_image_color, (x-200,y-200),(x+w+200, y+h+200),(255, 0, 0),4)
                cv.drawContours(interest_image_color, contours, -1, (0, 255, 0), 4)
                cv.drawContours(interest_image_color, biggest_contour, -1, (0, 0, 255), 10)

                # Extracting just biggest contour mask from color image and calculating RGB share.
                masked_contour_image = cv.bitwise_and(interest_image_color, interest_image_color, mask = dilation_image)
                black_pixels = np.sum(masked_contour_image == 0)
                avg_color_per_row = np.average(masked_contour_image, axis=0)
                avg_color = np.average(avg_color_per_row, axis=0)
                avg_color_sum = sum(avg_color)
                # print("B: " + str(round(avg_color[0]/avg_color_sum, 2)) + " G: " + str(round(avg_color[1]/avg_color_sum, 2)) + " R: " + str(round(avg_color[2]/avg_color_sum, 2)))

            color = ["b", "g", "r"]
            for i,col in enumerate(color):
                hist = cv.calcHist([interest_image_color], [i], binary_image, [256], [0, 256])
                plt.plot(hist, color = col)
                plt.xlim([0, 256])
            #plt.show()

            # Display each image with the calculated rectangle to see results.
            interest_image_color = cv.resize(interest_image_color, (1200, 700))
            sequence_images_modified.append(interest_image_color)
            #cv.imshow("Sequence 1 / Image " + str(i+1) + " - Remaining contour centroids after binarization and erosion", interest_image_color)
            cv.imshow("image", interest_image_color)
            cv.waitKey(0)
            cv.destroyAllWindows()

    # If image does not have sequence.
    else:
        print("No sequence could be detected for this image")

