from ultralytics import YOLO
import cv2
import numpy as np
import sys
from PIL import Image, ImageChops
import os
from pathlib import Path
import shutil

home =os.getcwd()


# Get the current directory (location of the script)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Navigate back to the parent directory
home_dir = os.path.dirname(current_dir)

path_raw = home_dir+'/Images/'
#file_name = 'us.jpg'
input_dir = sys.argv[2]
output_dir = int(input_dir) + 1
file_name = sys.argv[1]
full_filename = path_raw+file_name

#Clear output_dir
dir_to_empty = Path(home_dir+'/PipelineImages/'+str(output_dir))
[shutil.rmtree(item) if item.is_dir() else item.unlink() for item in dir_to_empty.glob('*') if item.exists()]

# [p.unlink() for p in Path(home_dir+'/PipelineImages/'+str(output_dir)).glob('*')]

#load Image
image = cv2.imread(full_filename)

#BGR to RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Load a model
model = YOLO('yolov8s-seg.pt')  # load an official model

# Predict with the model
results = model(full_filename)  # predict on an image

text_file_path = home+"/Thesis/dragApp/image_positions.txt"
# Open the file in write mode to empty its content
with open(text_file_path, "w") as file:
    pass  # Empty the file


def add_padding_with_transparency(image, mask, padding_percentage):
    img_width, img_height = image.size
    pad_x = int(img_width * padding_percentage)
    pad_y = int(img_height * padding_percentage)
    
    # Create a new image with transparent padding
    padded_image = Image.new("RGBA", (img_width + 2*pad_x, img_height + 2*pad_y))
    padded_image.paste(image, (pad_x, pad_y))
    
    # Create a new mask with white padding
    padded_mask = Image.new("L", (img_width + 2*pad_x, img_height + 2*pad_y), color=0)
    padded_mask.paste(mask, (pad_x, pad_y))
    
    return padded_image, padded_mask


def crop_image_with_mask(image, mask, crop_dim):
    
#     # Perform element-wise multiplication between the image and the mask; This will actually remove BG
#     image = image * mask[:, :, np.newaxis]

#     #BGR to RGB
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

    #getting cropping dimensions in separate variables
    x1,y1,x2,y2 = crop_dim
    
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
    
    image_crop = image.crop((x1, y1, x2, y2))
    mask_crop = mask.crop((x1, y1, x2, y2))
    
    return image_crop, mask_crop

bild_PIL = Image.open(full_filename).convert('RGBA')
# Mask generation from the transparent region of the image; Create a mask image with white pixels where alpha is transparent
alpha_mask = Image.new('L', bild_PIL.size)
alpha_mask.putdata([0 if a == 0 else 255 for r, g, b, a in bild_PIL.getdata()])
alpha_mask = np.array(alpha_mask)

i=0
bg_mask = np.zeros(results[0].orig_shape,dtype='uint8')

bg_box = np.zeros(results[0].orig_shape,dtype='uint8') #bild_PIL is to only get the shape of the original image here
# bg_box[:, :] = 255

if results[0].masks is not None:
    for mask in results[0].masks.masks.cpu().numpy():
        
        mask[mask==1] = 255
        #mask = np.moveaxis(mask, 0, -1)
        mask = mask.astype(dtype='uint8')
        #Resizing mask
        mask = cv2.resize(mask, tuple(reversed(results[0].orig_shape)), interpolation =cv2.INTER_CUBIC)
        masked = cv2.bitwise_and(image_rgb,image_rgb, mask=mask)
        instance_name = model.names[results[0].boxes.boxes[i][-1].cpu().numpy().item()]

        mask = cv2.bitwise_and(alpha_mask, mask)
        
        image_crop, mask_crop = crop_image_with_mask(masked, mask, results[0].boxes.boxes[i][:4].cpu().numpy() )

        #Getting the coordinates of the left top corner of the bounding box
        x1,y1, x2,y2 = results[0].boxes.boxes[i][:4].cpu().numpy()
        # Convert them to integers
        y1, y2, x1, x2 = map(int, [y1, y2, x1, x2])
        
        # Add padding to image and mask
        padded_image, padded_mask = add_padding_with_transparency(image_crop, mask_crop.convert("L"), 0.2)
        
        #save mask for inpainting
        cv2.imwrite(home_dir+'/PipelineImages/'+str(output_dir)+'/{}_segemented_image_invmask_[{}]_{}.png'.format(file_name.split(".")[0],instance_name,i), cv2.bitwise_not(np.array(padded_mask)))
        
        # Save the transparent image
        padded_image.save(home_dir+'/PipelineImages/'+str(output_dir)+'/{}_segemented_image_[{}]_{}.png'.format(file_name.split(".")[0],instance_name,i), "PNG")

        #Adjusting bb coordinates due to padding 
        x_1 = x1 - int(image_crop.size[0]*0.2)
        y_1 = y1 - int(image_crop.size[1]*0.2)

        # Open a text file for writing (you can specify the file path)
        with open(text_file_path, "a") as file:
            # Write lines to the file one by one
            file.write(home+"/Thesis/PipelineImages/5/{}_segemented_image_[{}]_{}.png {} {}\n".format(file_name.split(".")[0],instance_name,i,x_1,y_1))
            
    #     cv2.imwrite(home_dir+'/PipelineImages/'+str(output_dir)+'/{}_segemented_image_{}_[{}].jpg'.format(file_name.split(".")[0],i, instance_name), masked)
        i = i+1
        bg_mask = cv2.bitwise_or(bg_mask,mask)
        #For cropping BB from the background mask
        bg_box[y1:y2, x1:x2] = 255
            


bg_mask = cv2.bitwise_not(bg_mask)
bg_mask = cv2.bitwise_and(alpha_mask, bg_mask)
# background = cv2.bitwise_and(image_rgb, image_rgb, mask = bg_mask)

bg_box = cv2.bitwise_not(bg_box)
bg_box = cv2.bitwise_and(alpha_mask, bg_box)
background = cv2.bitwise_and(image_rgb, image_rgb, mask = bg_box)

#Full size of the image as x1, y1, x2, y2
full_size = np.array([0,0,image.shape[:2][1],image.shape[:2][0]])
background_transparent, _ = crop_image_with_mask(background, bg_box, full_size)

# cv2.imwrite(home_dir+'/PipelineImages/'+str(output_dir)+'/{}_background.jpg'.format(file_name.split(".")[0]), background)
background_transparent.save(home_dir+'/PipelineImages/'+str(output_dir)+'/{}_background.png'.format(file_name.split(".")[0]), "PNG")

cv2.imwrite(home_dir+'/PipelineImages/'+str(output_dir)+'/{}_background_invmask.png'.format(file_name.split(".")[0]), cv2.bitwise_not(bg_box))

# Save background positions 
with open(text_file_path, "a") as file:
    # Write lines to the file one by one
    file.write(home+"/Thesis/PipelineImages/5/{}_background.png {} {}".format(file_name.split(".")[0],0,0))


