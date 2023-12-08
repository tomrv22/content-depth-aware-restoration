#Deeplabv3 ------------------------------------------------------
import cv2
import torch
from PIL import Image
from torchvision import transforms
from matplotlib import pyplot as plt

def load_model():
  model = torch.hub.load('pytorch/vision:v0.6.0', 'deeplabv3_resnet101', pretrained=True)
  model.eval()
  return model

def make_transparent_foreground(pic, mask):
  # split the image into channels
  b, g, r = cv2.split(np.array(pic).astype('uint8'))
  # add an alpha channel with and fill all with transparent pixels (max 255)
  a = np.ones(mask.shape, dtype='uint8') * 255
  # merge the alpha channel back
  alpha_im = cv2.merge([b, g, r, a], 4)
  # create a transparent background
  bg = np.zeros(alpha_im.shape)
  # setup the new mask
  new_mask = np.stack([mask, mask, mask, mask], axis=2)
  # copy only the foreground color pixels from the original image where mask is set
  foreground = np.where(new_mask, alpha_im, bg).astype(np.uint8)

  return foreground

def remove_background(model, input_image):
  input_image = input_image.convert("RGB")
  preprocess = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
  ])

  input_tensor = preprocess(input_image)
  input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

  # move the input and model to GPU for speed if available
  if torch.cuda.is_available():
      input_batch = input_batch.to('cuda')
      model.to('cuda')

  with torch.no_grad():
      output = model(input_batch)['out'][0]
  output_predictions = output.argmax(0)

  # create a binary (black and white) mask of the profile foreground
  mask = output_predictions.byte().cpu().numpy()
  background = np.zeros(mask.shape)
  bin_mask = np.where(mask, 255, background).astype(np.uint8)

  foreground = make_transparent_foreground(input_image ,bin_mask)

  return foreground, bin_mask

#------------------------------------------------------------------------------

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
import shutil

# Get the current directory (location of the script)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Navigate back to the parent directory
home_dir = os.path.dirname(current_dir)

input_dir = sys.argv[1]
output_dir = int(input_dir) + 1

#Clear output_dir
from pathlib import Path
dir_to_empty = Path(home_dir+'/PipelineImages/'+str(output_dir))
[shutil.rmtree(item) if item.is_dir() else item.unlink() for item in dir_to_empty.glob('*') if item.exists()]

# [p.unlink() for p in Path(home_dir+'/PipelineImages/'+str(output_dir)).glob('*')]

folder_path = home_dir+'/PipelineImages/'+str(input_dir)+'/'
all_files = os.listdir(folder_path)

deeplab_model = load_model()


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
def dilate_binary_mask(mask, iterations):
    if iterations==0:
        dilated_mask = np.array(mask, dtype=bool)
    else:
        structuring_element = np.array([[1, 1, 1],
                                        [1, 1, 1],
                                        [1, 1, 1]], dtype=np.uint8)  # 3x3 structuring element

        dilated_mask = ndimage.binary_dilation(mask, structure=structuring_element, iterations=iterations)
    return dilated_mask



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
    dilated_mask = dilate_binary_mask(mask_image, iterations=3)
    
    dilated_image_mask = Image.fromarray(dilated_mask)
    image = image.resize((512, 512))
    mask = dilated_image_mask.resize((512, 512))


    label = re.search(r'\[(.*?)\]', file_name)
    if label:
        prompt = "one "+label.group(1)+"; plain background; high resolution; no other objects; sharp edges"
    else:
        prompt = "plain background; high resolution; no other objects; sharp edges"

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

    #Remove background using Deeplabv3
    if 'background' not in file_name:
        foreground, bin_mask = remove_background(deeplab_model, images[0].resize(size))

        Image.fromarray(foreground).save(home_dir+'/PipelineImages/'+str(output_dir)+'/{}_regenerated.png'.format(file_name))
    else:
        images[0].resize(size).save(home_dir+'/PipelineImages/'+str(output_dir)+'/{}_regenerated.png'.format(file_name))

    #images[0].resize(size).save(home_dir+'/PipelineImages/'+str(output_dir)+'/{}_regenerated.jpg'.format(file_name))