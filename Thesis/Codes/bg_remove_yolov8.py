from ultralytics import YOLO
import cv2
import numpy as np
import sys
from PIL import Image
import os
import re
import shutil

# Get the current directory (location of the script)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Navigate back to the parent directory
home_dir = os.path.dirname(current_dir)

input_dir = sys.argv[1]
output_dir = int(input_dir)+1

#Clear output_dir
from pathlib import Path
dir_to_empty = Path(home_dir+'/PipelineImages/'+str(output_dir))
[shutil.rmtree(item) if item.is_dir() else item.unlink() for item in dir_to_empty.glob('*') if item.exists()]

# [p.unlink() for p in Path(home_dir+'/PipelineImages/'+str(output_dir)).glob('*')]

# Folder path where the images are located
folder_path = home_dir+'/PipelineImages/'+ input_dir

#To find the class label from the filename
find_text_between_brackets = lambda s: re.search(r'\[(.*?)\]', s).group(1) if re.search(r'\[(.*?)\]', s) else None


#Crop instances
def crop_image_with_mask(image, mask):
    
    # Perform element-wise multiplication between the image and the mask; This will actually remove BG
    image = image * mask[:, :,np.newaxis]
    
    # Load the image and the mask into PIL Image formats from arrays
    image = Image.fromarray(image)
    mask = Image.fromarray(mask)  # Convert to grayscale (L) if needed

    # Add an alpha channel if the image doesn't have one (mode='RGBA')
    image = image.convert("RGBA")

    # Apply the mask to the alpha channel
    data = image.getdata()
    new_data = []
    for i, pixel in enumerate(data):
        if mask.getdata()[i] == 0:  # Check if the mask pixel is black (0)
            new_data.append((pixel[0], pixel[1], pixel[2], 0))  # Set alpha to 0 (fully transparent)
        else:
            new_data.append(pixel)  # Keep the original pixel if not black in the mask

    image.putdata(new_data)
    
    return image


# Load a model
model = YOLO('yolov8s-seg.pt')  # load an official model

# Loop through all files in the folder
for filename in os.listdir(folder_path):
    
    #percentage score for segmenting instance; flag; done to get the max. percentage for similar classes
    score_pct = 0
    
    # Check if the file is an image (optional)
    if filename.endswith(('.png', '.jpg', '.jpeg')) and ('background' not in filename):
        # Create the file's absolute path
        image_path = os.path.join(folder_path, filename)

        # Apply YOLOv8 instance segmentation
        results = model(image_path)
        
        # Read the image for cropping by segment
        bild = cv2.imread(image_path)

        #BGR to RGB
        bild = cv2.cvtColor(bild, cv2.COLOR_BGR2RGB)
        if results[0].masks is not None:
            for i in range(len(results[0].masks.masks)):
                
                mask_array = results[0].masks.masks.cpu().numpy()[i,:,:]
                mask_array = cv2.resize(mask_array, tuple(reversed(bild.shape[:2])), interpolation =cv2.INTER_CUBIC)
                mask_array = mask_array.astype(bool)
                class_label = model.names[results[0].boxes.boxes.cpu().numpy()[i][5]]
    #             print(class_label)
                
                if (class_label == find_text_between_brackets(filename)) and (results[0].boxes.boxes[i][4]> score_pct) :
                    score_pct = results[0].boxes.boxes[i][4]
                    # Crop the image based on the mask
                    cropped_image = crop_image_with_mask(bild, mask_array)
                       
                    # Save the cropped image
                    cropped_image.save(home_dir+'/PipelineImages/'+str(output_dir)+'/{}_segemented_image_{}.png'.format(filename.split(".")[0], i), "PNG")
                
    elif 'background' in filename:
        
        bgimage_path = os.path.join(folder_path, filename)
        background = Image.open(bgimage_path)
        
        background.save(home_dir+'/PipelineImages/'+str(output_dir)+'/{}.png'.format(filename.split(".")[0]), "PNG")

#Renaming for dragAPP
directory_path = home_dir+'/PipelineImages/'+str(output_dir)

# Check if the directory exists
if os.path.exists(directory_path) and os.path.isdir(directory_path):
    # List all files in the directory
    for filename in os.listdir(directory_path):
        old_file_path = os.path.join(directory_path, filename)
        
        # Use regular expressions to find the first number after  in the filename
        # match_number = re.search(r'\d', filename)

        # # Use a regular expression to match the pattern and extract the parts
        # pattern = r'(\d.*?_\[\w+\]_\d)(_.*?\..*$)'
        # match = re.match(pattern, filename)
        
        if 'background' not in filename:
            new_filename = filename[:filename.find(']')+3]+'.png'

            new_file_path = os.path.join(directory_path, new_filename)
            # Rename the file
            os.rename(old_file_path, new_file_path)
        elif 'background' in filename:
            # # Use a regular expression to match 'background' and everything after it
            # pattern = r'(.*?background).*$'
            # new_filename = re.sub(pattern, r'\1', filename)+'.png'

            # Use a regular expression to match 'background' and everything between it and '.'
            pattern = r'(.*?background).*?(\..*$)'
            new_filename = re.sub(pattern, r'\1\2', filename)

            new_file_path = os.path.join(directory_path, new_filename)
            # Rename the file
            os.rename(old_file_path, new_file_path)
            
        else:
            print(f"No number found in the filename: {filename}")
else:
    print("The specified directory does not exist.")
