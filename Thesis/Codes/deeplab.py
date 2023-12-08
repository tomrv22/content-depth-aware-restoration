import cv2
import torch
from PIL import Image
from torchvision import transforms
from matplotlib import pyplot as plt
import numpy as np
import sys
import os
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

        bild_PIL = Image.open(image_path).convert('RGBA')
        # Mask generation from the transparent region of the image; Create a mask image with white pixels where alpha is transparent
        alpha_mask = Image.new('L', bild_PIL.size)
        alpha_mask.putdata([0 if a == 0 else 255 for r, g, b, a in bild_PIL.getdata()])
        alpha_mask = np.array(alpha_mask)
        
        #Saving as transaprent using PIL.Image
        bgr_foreground = Image.fromarray(foreground)
        bgr_foreground.save(home_dir+'/PipelineImages/'+str(output_dir)+'/{}'.format(filename), "PNG")
        
        # #RGB to BGR
        # bgr_foreground = cv2.cvtColor(foreground, cv2.COLOR_RGB2BGR)
        # cv2.imwrite(home_dir+'/PipelineImages/'+str(output_dir)+'/{}'.format(filename), bgr_foreground)

        #bitwise NOT ----> inverts the binary color
        # Perform bitwise OR operation on the masks
        bin_mask = cv2.bitwise_and(alpha_mask, bin_mask)
        inv_mask = cv2.bitwise_not(bin_mask)
        cv2.imwrite(home_dir+'/PipelineImages/'+str(output_dir)+'/{}_invmask.png'.format(filename.split(".")[0]), inv_mask)
                
    elif 'background' in filename:
        
        bgimage_path = os.path.join(path_crop, filename)
        background = Image.open(bgimage_path)
               
        # Create a mask from the alpha channel of the background image
        alpha_channel = background.split()[3]  # Get the alpha channel
        binary_mask = alpha_channel.point(lambda p: 0 if p > 0 else 255)  # Create binary mask
        
        background.save(home_dir+'/PipelineImages/'+str(output_dir)+'/{}.png'.format(filename.split(".")[0]), "PNG")
        binary_mask.save(home_dir+'/PipelineImages/'+str(output_dir)+'/{}_invmask.png'.format(filename.split(".")[0]), "PNG")
        