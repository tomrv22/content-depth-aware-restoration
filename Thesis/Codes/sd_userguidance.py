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
# model_path = "runwayml/stable-diffusion-inpainting"
model_path = "stabilityai/stable-diffusion-2-inpainting"

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
import shutil

# Get the current directory (location of the script)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Navigate back to the parent directory
home_dir = os.path.dirname(current_dir)

input_dir = sys.argv[1]
guide_scale = float(sys.argv[2])
dilation_iter = int(sys.argv[3])
output_dir = int(input_dir) + 1

#Clear output_dir
from pathlib import Path
dir_to_empty = Path(home_dir+'/PipelineImages/'+str(output_dir))
[shutil.rmtree(item) if item.is_dir() else item.unlink() for item in dir_to_empty.glob('*') if item.exists()]

# [p.unlink() for p in Path(home_dir+'/PipelineImages/'+str(output_dir)).glob('*')]


folder_path = home_dir+'/PipelineImages/'+str(input_dir)+'/'
all_files = os.listdir(folder_path)


def generate_pairs(input_list):
    pairs = []
    n = len(input_list)
    
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((input_list[i], input_list[j]))
    
    return pairs

#generate full pairs:
pairs = generate_pairs(all_files)

filtered_pairs = []
for pair in pairs:
    name1 = pair[0].replace('_invmask',"")
    name2 = pair[1].replace('_invmask',"")
    if name1 == name2:
        filtered_pairs.append(pair)

#For dilating the image mask
def morphological_operation(mask, iterations):
    structuring_element = np.array([[1, 1, 1],
                                    [1, 1, 1],
                                    [1, 1, 1]], dtype=np.uint8)  # 3x3 structuring element

    if iterations > 0:
        result_mask = ndimage.binary_dilation(mask, structure=structuring_element, iterations=iterations)
    elif iterations < 0:
        result_mask = ndimage.binary_erosion(mask, structure=structuring_element, iterations=abs(iterations), border_value=1)
    elif iterations == 0:
        result_mask = np.array(mask, dtype=np.uint8)   # No change for iterations = 0

    return result_mask


for pair in filtered_pairs:
    if '_invmask' in pair[0]:
        mask_name = pair[0]
        file_name = pair[1]
    else:
        mask_name = pair[1]
        file_name = pair[0]
    
    size = Image.open(folder_path+file_name).size
    image = Image.open(folder_path+file_name)
    mask_image = Image.open(folder_path+mask_name)
    
    #dilating mask
    dilated_mask = morphological_operation(mask_image, iterations=dilation_iter)
    
    dilated_image_mask = Image.fromarray(dilated_mask)
    image = image.resize((512, 512))
    mask = dilated_image_mask.resize((512, 512))
    
    label = re.search(r'\[(.*?)\]', file_name)
    if label:
        prompt = "one "+label.group(1)+"; plain background; high resolution; no other objects; sharp edges"
    else:
        # prompt = "empty"
        prompt = "plain background; high resolution; no other objects; sharp edges"
        # prompt = "Remove objects; blend seamlessly with the surroundings"

    guidance_scale=guide_scale
    num_samples = 3
    generator = torch.Generator(device="cuda").manual_seed(3) # change the seed to get different results

    images = pipe(
        prompt=prompt,
        image=image,
        mask_image= mask,
        guidance_scale=guidance_scale,
        generator=generator,
        num_images_per_prompt=num_samples,
    ).images


    images[0].resize(size).save(home_dir+'/PipelineImages/'+str(output_dir)+'/{}_regenerated.png'.format(file_name.split('.')[0]))