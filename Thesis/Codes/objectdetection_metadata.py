from ultralytics import YOLO
from PIL import Image
import numpy as np
import sys
import os
home = os.getcwd()

# Load a model
model = YOLO('yolov3u.pt')  # load an official model

folder_path = home+'/Thesis/TestingDir/GroundTruth/'


ObjDetection_textfile = folder_path+"ObjDetection_textfile.txt"
# Open the file in write mode to empty its content
with open(ObjDetection_textfile, "w") as file:
    pass  # Empty the file

# Loop through all files in the folder
for filename in os.listdir(folder_path):
    
    # Check if the file is an image (optional)
    if filename.endswith(('.png', '.jpg', '.jpeg')):
        
        # Predict with the model
        results = model(folder_path+filename)  # predict on an image
        
        # Open a text file for writing append mode
        with open(ObjDetection_textfile, "a") as file:
            file.write(filename+"\t")
        
        #Crop and save to directory
        for i in range(len(results[0].boxes.boxes)):
            box = np.array(results[0].boxes.boxes[i].cpu())
            
            class_label = model.names[box[5]]
            score = box[4]
            
            # Open a text file for writing append mode to add class_labels and scores
            with open(ObjDetection_textfile, "a") as file:
                file.write(class_label+"-"+str(score)+"\t")
         
        
        # Open a text file for writing append mode to switch to next line
        with open(ObjDetection_textfile, "a") as file:
            file.write("\n")

