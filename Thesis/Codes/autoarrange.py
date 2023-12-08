from PIL import Image
import os

home = os.getcwd()

# Initialize variables to track the canvas size and the background image path.
canvas_width, canvas_height = 0, 0
background_image_path = None

# Create a list to store image paths and their respective positions.
images = []

# Read the input text file.
with open(home+'/Thesis/dragApp/image_positions.txt', 'r') as file:
    for line in file:
        parts = line.rsplit(None, 2)
        image_path = parts[0]
        x, y = int(parts[1]), int(parts[2])

        # Check if the image file exists at the specified location.
        if not os.path.exists(image_path):
            print(f"Image file not found: {image_path}. Skipping...")
            continue

        # Check if the image path contains the term 'background'.
        if 'background' in image_path:
            background_image_path = image_path
        else:
            images.append((image_path, x, y))

        # Update canvas size based on the image dimensions.
        image = Image.open(image_path)
        width, height = image.size
        canvas_width = max(canvas_width, x + width)
        canvas_height = max(canvas_height, y + height)

# Create the canvas with the calculated size and a transparent background.
canvas = Image.new('RGBA', (canvas_width, canvas_height))

# Load the background image onto the canvas with alpha transparency.
if background_image_path:
    background_image = Image.open(background_image_path)
    canvas.paste(background_image, (0, 0))

# Load the other images onto the canvas with alpha transparency.
for image_path, x, y in images:
    image = Image.open(image_path)
    canvas.paste(image, (x, y), image)

# Save the canvas as a single image.
output_filename = os.path.splitext(os.path.basename(background_image_path))[0].replace('_background', '') + '.png'
output_path = home+'/Thesis/PipelineImages/6/' + output_filename
canvas.save(output_path)

print(f"Canvas saved as '{output_filename}'.")
