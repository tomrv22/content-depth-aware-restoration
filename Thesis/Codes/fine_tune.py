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


images[0].resize(size).save(home_dir+'/PipelineImages/'+str(output_dir)+'/{}'.format(image_2_tune))