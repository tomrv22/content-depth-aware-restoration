import os
import shutil
import sys
from pathlib import Path

# Get the current directory (location of the script)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Navigate back to the parent directory
home_dir = os.path.dirname(current_dir)+'/PipelineImages/'

input_dir = sys.argv[1]
output_dir = int(input_dir) + 1

#Clear output_dir
dir_to_empty = Path(home_dir+str(output_dir))
[shutil.rmtree(item) if item.is_dir() else item.unlink() for item in dir_to_empty.glob('*') if item.exists()]

# [p.unlink() for p in Path(home_dir+'/PipelineImages/'+str(output_dir)).glob('*')]

#Writing the full path
input_dir = home_dir+str(input_dir)+'/'
output_dir = home_dir+str(output_dir)+'/'


# Ensure the source directory exists
if not os.path.exists(input_dir):
    print(f"Input directory '{input_dir}' does not exist.")
    sys.exit(1)

# Ensure the destination directory exists; create it if it doesn't
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# List all image files in the source directory
image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif'))]

# Copy image files to the output_dir
for image_file in image_files:
    source_path = os.path.join(input_dir, image_file)
    destination_path = os.path.join(output_dir, image_file)
    
    # Use shutil.copy to copy the file
    shutil.copy(source_path, destination_path)

#print(f"{len(image_files)} image files copied from '{input_dir}' to '{output_dir}'.")
