import cv2
import torch
from PIL import Image
from torchvision import transforms
from matplotlib import pyplot as plt
import numpy as np
import sys
import os
import re
import shutil

# Get the current directory (location of the script)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Navigate back to the parent directory
home_dir = os.path.dirname(current_dir)

input_dir = sys.argv[1]
output_dir = int(input_dir) + 1
path_crop = home_dir+'/PipelineImages/'+str(input_dir)

#Clear output_dir
from pathlib import Path
dir_to_empty = Path(home_dir+'/PipelineImages/'+str(output_dir))
[shutil.rmtree(item) if item.is_dir() else item.unlink() for item in dir_to_empty.glob('*') if item.exists()]

# [p.unlink() for p in Path(home_dir+'/PipelineImages/'+str(output_dir)).glob('*')]


import torch
import torch.hub
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms

def load_model():
    model = torch.hub.load('pytorch/vision:v0.6.0', 'deeplabv3_resnet101', pretrained=True)
    model.eval()
    if torch.cuda.is_available():
        model.to('cuda')
    return model

def make_transparent_foreground(pic, mask):
    pic = np.array(pic).astype(np.uint8)
    a = np.ones(mask.shape, dtype=np.uint8) * 255
    alpha_im = cv2.merge([pic[:, :, 0], pic[:, :, 1], pic[:, :, 2], a], 4)
    bg = np.zeros(alpha_im.shape, dtype=np.uint8)
    new_mask = np.stack([mask, mask, mask, mask], axis=2)
    foreground = np.where(new_mask, alpha_im, bg).astype(np.uint8)
    return foreground

def remove_background(model, input_file):
    input_image = Image.open(input_file).convert("RGB")
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)

    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')

    with torch.no_grad():
        output = model(input_batch)['out'][0]
    output_predictions = output.argmax(0)

    mask = output_predictions.byte().cpu().numpy()
    background = np.zeros(mask.shape, dtype=np.uint8)
    bin_mask = np.where(mask, 255, background).astype(np.uint8)

    foreground = make_transparent_foreground(input_image, bin_mask)

    return foreground, bin_mask

deeplab_model = load_model()


# Loop through all files in the folder
for filename in os.listdir(path_crop):
    
    # Check if the file is an image (optional)
    if filename.endswith(('.png', '.jpg', '.jpeg')) and ('background' not in filename):
        
        # Create the file's absolute path
        image_path = os.path.join(path_crop, filename)
        foreground, bin_mask = remove_background(deeplab_model, image_path)
        
        #Saving as transaprent using PIL.Image
        bgr_foreground = Image.fromarray(foreground)
        bgr_foreground.save(home_dir+'/PipelineImages/'+str(output_dir)+'/{}_bgremoved.png'.format(filename.split('.')[0]), "PNG")
        
        # #RGB to BGR
        # bgr_foreground = cv2.cvtColor(foreground, cv2.COLOR_RGB2BGR)
        # cv2.imwrite(home_dir+'/PipelineImages/'+str(output_dir)+'/{}'.format(filename), bgr_foreground)
        
                
    elif 'background' in filename:
        
        bgimage_path = os.path.join(path_crop, filename)
        background = Image.open(bgimage_path)
               
        # # Create a mask from the alpha channel of the background image
        # alpha_channel = background.split()[3]  # Get the alpha channel
        # binary_mask = alpha_channel.point(lambda p: 0 if p > 0 else 255)  # Create binary mask
        
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