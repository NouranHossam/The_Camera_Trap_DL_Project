import os
import glob
import shutil
import cv2 as cv
import numpy as np
import math
import imutils
import json
from matplotlib import pyplot as plt
from PIL import Image
from datetime import datetime, timedelta

# CAMERA TRAP CHALLENGE WS2022/23 - STUDY PROJECT:
# The script will process images in the following order: finding coherent sequences,
# locating animals in in the images, classifying animals into different categories.

# Load all parameters from config file.
with open("config.json", "r") as jsonfile:
    config = json.load(jsonfile)

############################## TASK 1 - CREATE BDD DATASET ##############################

# Folder settings
destination_dir = "BDD"
background_dir = "backgrounds"
average_dir = "average_backgrounds"
source_dir = ["dama_dama_damhirsch_deer/dama_dama_damhirsch", "meles_meles_dachs_badger/meles_meles_dachs"]

# Create directories.
if not os.path.exists(destination_dir):
    os.makedirs(destination_dir)
if not os.path.exists(background_dir):
    os.makedirs(background_dir)
if not os.path.exists(average_dir):
    os.makedirs(average_dir)

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

print("Destination folder with animals combined: " + str(len(os.listdir(destination_dir))) + " images.\n" +
      "Background folder with empty images combined: " + str(len(os.listdir(background_dir))) + " images.")


############################## TASK 2 - IDENTIFY IMAGE SEQUENCES ##############################

'''

# Go through background images and look for similar images.
background_images = os.listdir(background_dir)
images_to_check = background_images.copy()
all_backgrounds = []

if not os.path.exists(average_dir):
    for i in range(len(background_images)):
        if (background_images[i] in images_to_check):
            print("Similarities for background image " + str(i))
            current_image_path = os.path.join(background_dir, background_images[i])
            current_image = cv.imread(current_image_path, cv.IMREAD_GRAYSCALE)
            # Update status.
            images_to_check[i] = "checked"
            same_background_images = [(background_images[i])]
            for j in (os.listdir(background_dir)):
                if (j in images_to_check):
                    other_image_path = os.path.join(background_dir, j)
                    other_image = cv.imread(other_image_path, cv.IMREAD_GRAYSCALE)

                    # Calculate RMSE for similarity.
                    MSE = np.square(np.subtract(current_image, other_image)).mean()
                    RMSE = math.sqrt(MSE)

                    if (RMSE < config["rmse_threshold_empty_background"]):
                        if not (j == background_images[i]):
                            same_background_images.append(j)
                            images_to_check[os.listdir(background_dir).index(j)] = "found"

            # For each identified background sequence, print image names and sequence size.
            print(str(len(same_background_images)) + " images of same background found: " + str(same_background_images))
            all_backgrounds.append(same_background_images)

# Calculate average background images.
for backgrounds in all_backgrounds:

    current_background_images = []

    for image in backgrounds:
        # To save image in sequence array.
        current_image_path = os.path.join(background_dir, image)
        current_image = cv.imread(current_image_path, cv.IMREAD_GRAYSCALE)
        current_background_images.append(current_image)

    background_average = current_background_images[0]

    for image in range(len(current_background_images)):
        # First image already included above.
        if image == 0:
            pass
        else:
            # Check again how this exactly works - three lines copied from Stack Overflow.
            alpha = 1.0 / (image + 1)
            beta = 1.0 - alpha
            background_average = cv.addWeighted(current_background_images[image], alpha, background_average, beta, 0.0)
    # Save average images in folder.
    cv.imwrite(average_dir + "/" + str(backgrounds[0]), background_average)


# From here on, work with new folder as dataset
image_filenames = os.listdir(destination_dir)
average_background_filenames = os.listdir(average_dir)

if not os.path.exists("sorted"):
    os.makedirs("sorted")

if not os.path.exists("sorted/unmatched"):
    os.makedirs("sorted/unmatched")

for image in image_filenames:
    current_image_path = os.path.join(destination_dir, image)
    current_image = cv.imread(current_image_path, cv.IMREAD_GRAYSCALE)

    lowest_rmse = 10
    similar_background = "tbd"

    for background in average_background_filenames:

        if not os.path.exists("sorted/" + str(background)):
            os.makedirs("sorted/" + str(background))

        current_average_image_path = os.path.join(average_dir, background)
        current_average_image = cv.imread(current_average_image_path, cv.IMREAD_GRAYSCALE)

        # Calculate RMSE for similarity.
        MSE = np.square(np.subtract(current_image, current_average_image)).mean()
        RMSE = math.sqrt(MSE)

        if RMSE < lowest_rmse:
            lowest_rmse = RMSE
            similar_background = background

    if lowest_rmse < config["rmse_threshold_with_animal"]:
        cv.imwrite("sorted/" + str(similar_background) + "/" + image, current_image)
    else:
        cv.imwrite("sorted/unmatched/" + image, current_image)

'''

#From here on, work with new folder as dataset
image_filenames = os.listdir(destination_dir)

'''
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

        # Extract camera name
        height, width = current_image.shape
        current_camera_name = current_image[int(height * 0.98):, :int(width * 0.2)]

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

                # Make use of camera-name in bottom left of image.
                other_camera_name = other_image[int(height * 0.98):, :int(width * 0.2)]
                camera_name_difference = other_camera_name.copy()
                cv.absdiff(current_camera_name, other_camera_name, camera_name_difference)
                white_pixels = np.sum(camera_name_difference == 255)

                # Calculate RMSE.
                MSE = np.square(np.subtract(current_image, other_image)).mean()
                RMSE = math.sqrt(MSE)
                print("Similarity of " + str(RMSE) + " between images " + image + " / " + image_filenames[i])

                # Before & after tolerance can be adjusted for datetime similarity.
                before_tolerance = current_date_time - timedelta(minutes = config["time_tolerance"])
                after_tolerance = current_date_time + timedelta(minutes = config["time_tolerance"])

                # Here we can switch between using RMSE and Camera name for sequence identification.
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

'''


############################## TASK 3 - LOCATING ANIMAL IN IMAGE ##############################

# Load image sequences from text file to list again.
all_sequences = []

with open("all_sequences.txt") as file:
    for line in file:
        all_sequences.append(line.split())

#all_sequences = [["IMG_2547.JPG", "IMG_2548.JPG", "IMG_2549.JPG", "IMG_2550.JPG", "IMG_2551.JPG"], ["IMG_2079.JPG", "IMG_2080.JPG", "IMG_2081.JPG", "IMG_2082.JPG", "IMG_2083.JPG", "IMG_2084.JPG"]]

# Loop through all sequences.
for sequence in all_sequences:

    print(sequence)
    sequence = sequence
    sequence_images = []
    sequence_images_color = []

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
                cv.rectangle(interest_image_color, (x-200,y-200),(x+w+200, y+h+200),(128, 0, 0),2)
                cv.drawContours(interest_image_color, contours, -1, (0, 255, 0), 2)
                cv.drawContours(interest_image_color, biggest_contour, -1, (0, 0, 255), 5 )

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
            #cv.imshow("Sequence 1 / Image " + str(i+1) + " - Remaining contour centroids after binarization and erosion", interest_image_color)
            #cv.imshow("image", interest_image_color)
            #cv.waitKey(0)
            #cv.destroyAllWindows()

    # If image does not have sequence.
    else:
        print("No sequence could be detected for this image")

