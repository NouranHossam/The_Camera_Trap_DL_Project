from ultralytics import YOLO
import cv2 as cv
import os
  
#### INFO FOR REVIEWERS ####
# As we do not want to provide any project images
# publicly on GitHub, they were deleted from the project.
# For running the model on the images, just fill the 
# "BDD_test" folder with images for prediction.  
  
# Setting paths for model, test images and predictions.
model_path = "runs/detect/train5/weights/best.pt"
image_source_path = "BDD_tests"
image_destination_path = "runs/detect/predict"
image_information_path = "runs/detect/predict/labels"

# Running own YOLOv8 model trained on images. Model is 
# trained on > 1000 camera trap images with 4 classes.
#model = YOLO(model_path)
#results = model.predict(source= image_source_path, max_det = 1, conf=0.25, save=True, save_txt = True)


# Function to estimate animal proximity from camera using 
# bounding box area and center y coordinate.
def estimate_proximity(image, image_data):

    bb_area = float(image_data[3]) * float(image_data[4])
    center_y = float(image_data[2])
    bottom_y = float(image_data[4])/2 + center_y
    distance_by_bottom_y = (1 - bottom_y**2) * 15
    distance_final = distance_by_bottom_y * (1 - bb_area)

    return round(distance_final, 1)

# Looping through test images.
for x in range(40): #len(results)

    # Loading image with prediction from folder.
    image_name = os.listdir(image_destination_path)[x]
    image_path = os.path.join(image_destination_path, image_name)
    image = cv.imread(image_path, cv.IMREAD_COLOR)

    # Loading bounding box and class information from folder.
    image_info_name = image_name.split('.')[0]
    image_info_name = image_info_name + ".txt"
    image_info_path = os.path.join(image_information_path, image_info_name)

    # If there is a textfile with same name as the image, proceed.
    if (os.path.exists(image_info_path)):

        print("Animal with bounding box found. Distance predicted.")
        image_info = open(image_info_path, "r")
        image_data = image_info.read()
        image_data = image_data.replace("\n", " ").split(" ")
        image_info.close()

        # Calculate x and y coordinates for proximity estimation text (bottom left).
        width = image.shape[1]
        height = image.shape[0]
        bb_x = (float(image_data[1]) - float(image_data[3])/2) * width
        bb_y = (float(image_data[2]) + float(image_data[4])/2) * height
        text_position_bottom_left = (int(bb_x), int(bb_y + height/50))

        # Write estimation to image.
        proximity_estimation = estimate_proximity(image, image_data)
        cv.putText(image, str(proximity_estimation) + "m", text_position_bottom_left, cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)

    # If no textfile is found, therefore test image has no identified animal.
    else:
        print("No animal found. No distance prediction.")

    # Resize and show image on screen.
    image = cv.resize(image, (1200, 700))
    cv.imshow("image", image)
    cv.waitKey(0)
    cv.destroyAllWindows()
