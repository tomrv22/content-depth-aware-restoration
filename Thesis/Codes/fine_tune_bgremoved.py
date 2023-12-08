import inspect
from typing import List, Optional, Union

import numpy as np
import torch
from scipy import ndimage

import PIL
import gradio as gr
from diffusers import StableDiffusionInpaintPipeline

import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))


device = "cuda"
model_path = "runwayml/stable-diffusion-inpainting"

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
).to(device)

import os
import random
import re
from PIL import Image
import torch
import sys

# Get the current directory (location of the script)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Navigate back to the parent directory
home_dir = os.path.dirname(current_dir)

image_2_tune = sys.argv[1]
prompt = sys.argv[2]
input_dir = sys.argv[3]
output_dir = int(input_dir) #Same as input directory

folder_path = home_dir+'/PipelineImages/'+str(input_dir)+'/'

    
size = Image.open(folder_path+image_2_tune).size
mask_image = Image.new("RGB", size, "white")
image = Image.open(folder_path+image_2_tune)

image = image.resize((512, 512))
mask = mask_image.resize((512, 512))

guidance_scale=17
num_samples = 2
generator = torch.Generator(device="cuda").manual_seed(3) # change the seed to get different results

images = pipe(
    prompt=prompt,
    image=image,
    mask_image= mask,
    guidance_scale=guidance_scale,
    generator=generator,
    num_images_per_prompt=num_samples,
).images

tuned_image = images[0].resize(size)
#------------------------------------------------YOLO v8 FOR  BG REMOVAL------------------------------------------------------

from ultralytics import YOLO
import cv2
import numpy as np
import sys
from PIL import Image
import os
import re

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

if 'background' not in image_2_tune:
    
    # Apply YOLOv8 instance segmentation
    results = model(tuned_image)
    
    # Convert to OpenCV Image
    bild = np.array(tuned_image)
    
    #BGR to RGB
    # bild = cv2.cvtColor(bild, cv2.COLOR_BGR2RGB)
    
    for i in range(len(results[0].masks.masks)):
        
        mask_array = results[0].masks.masks.cpu().numpy()[i,:,:]
        mask_array = cv2.resize(mask_array, tuple(reversed(bild.shape[:2])), interpolation =cv2.INTER_CUBIC)
        mask_array = mask_array.astype(bool)
        class_label = model.names[results[0].boxes.boxes.cpu().numpy()[i][5]]
    #             print(class_label)
        
        # if (class_label == find_text_between_brackets(filename)) and (results[0].boxes.boxes[i][4]> score_pct) :
        score_pct = results[0].boxes.boxes[i][4]
        # Crop the image based on the mask
        cropped_image = crop_image_with_mask(bild, mask_array)
           
        # Save the cropped image
        cropped_image.save(home_dir+'/PipelineImages/'+str(output_dir)+'/{}'.format(image_2_tune), "PNG")
                
elif 'background' in image_2_tune:

    tuned_image.save(home_dir+'/PipelineImages/'+str(output_dir)+'/{}'.format(image_2_tune), "PNG")
    
