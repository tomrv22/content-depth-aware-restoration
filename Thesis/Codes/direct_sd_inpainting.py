import inspect
import random
import sys
from typing import List, Optional, Union
from PIL import Image
import os
home = os.getcwd()

import numpy as np
import torch

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

input_folder_path = home+'/Thesis/TestingDir/'
output_folder_path = home+'/Thesis/TestingDir/SdRestored/'

# Loop through all files in the folder
for image_filename in os.listdir(input_folder_path+'Distorted/'):
    
    # Check if the file is an image (optional)
    if image_filename.endswith(('.png', '.jpg', '.jpeg')):
        
        mask_filename = image_filename.split('.')[0]+'_masked.png'
        
        image = Image.open(input_folder_path+'Distorted/'+image_filename)
        mask = Image.open(input_folder_path+'DistortedMasks/'+mask_filename)
        size = image.size
        
        image = image.resize((512, 512))
        mask = mask.resize((512, 512))
        
        
        prompt = ""
        
        guidance_scale=10
        num_samples = 1
        generator = torch.Generator(device="cuda").manual_seed(59) # change the seed to get different results
        
        images = pipe(
            prompt=prompt,
            image=image,
            mask_image= mask,
            guidance_scale=guidance_scale,
            generator=generator,
            num_images_per_prompt=num_samples,
        ).images
        
        
        images[0].resize(size).save(output_folder_path+image_filename, 'PNG')