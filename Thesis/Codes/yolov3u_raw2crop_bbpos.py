from ultralytics import YOLO
from PIL import Image
import numpy as np
import sys
import os
import shutil

home =os.getcwd()

# Get the current directory (location of the script)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Navigate back to the parent directory
home_dir = os.path.dirname(current_dir)

# Load a model
model = YOLO('yolov3u.pt')  # load an official model

path_raw = home_dir+'/Images/'

input_dir = sys.argv[2]
output_dir = int(input_dir) + 1 
file_name = sys.argv[1]

#Clear output_dir
from pathlib import Path
dir_to_empty = Path(home_dir+'/PipelineImages/'+str(output_dir))
[shutil.rmtree(item) if item.is_dir() else item.unlink() for item in dir_to_empty.glob('*') if item.exists()]

# [p.unlink() for p in Path(home_dir+'/PipelineImages/'+str(output_dir)).glob('*')]

full_filename = path_raw+file_name
image = Image.open(full_filename)

# Predict with the model
results = model(full_filename)  # predict on an image

background = Image.open(full_filename)
mask = Image.new('L', background.size, 255)

#Create text file to store BB positions
text_file_path = home+"/Thesis/dragApp/image_positions.txt"
# Open the file in write mode to empty its content
with open(text_file_path, "w") as file:
    pass  # Empty the file

#Crop and save to directory
if results[0].boxes is not None:
    for i in range(len(results[0].boxes.boxes)):
        box = np.array(results[0].boxes.boxes[i].cpu())
        # Get the coordinates of the bounding box
        x1, y1, x2, y2 = box[:4]
        class_id = box[5]
        # Crop the image using the defined coordinates
        image = Image.open(full_filename)
        cropped_image = image.crop((x1, y1, x2, y2))
    
        # Create a mask image with transparency
        x1, y1, x2, y2 = [int(value) for value in [x1, y1, x2, y2]]
        mask.paste(0, (x1, y1, x2, y2))
    
        # Save the cropped image; Done using PNG format to retain the transparency
        #cropped_image.save(home_dir+'/PipelineImages/'+str(output_dir)+'/{}_cropped_object_[{}]_{}.jpg'.format(file_name.split(".")[0],model.names[class_id],i))
        cropped_image.save(home_dir+'/PipelineImages/'+str(output_dir)+'/{}_cropped_object_[{}]_{}.png'.format(file_name.split(".")[0],model.names[class_id],i), format='PNG')
        
        # Open a text file for writing (you can specify the file path)
        with open(text_file_path, "a") as file:
            # Write lines to the file one by one
            file.write("http://127.0.0.1:5000/image/{}_cropped_object_[{}]_{}.png {} {}\n".format(file_name.split(".")[0],model.names[class_id],i,x1,y1))
    

# Create a new RGBA image with the same size as the original
background = Image.new('RGBA', image.size)

# Paste the original image onto the new image using the mask to make the removed region transparent
background.paste(image, (0, 0), mask)

background.save(home_dir+'/PipelineImages/'+str(output_dir)+'/{}_background.png'.format(file_name.split(".")[0]))

# Save background positions 
with open(text_file_path, "a") as file:
    # Write lines to the file one by one
    file.write("http://127.0.0.1:5000/image/{}_background.png {} {}".format(file_name.split(".")[0],0,0))

