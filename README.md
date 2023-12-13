# content-aware depth-adaptive image-restoration

## Setup environment
The requirements.txt file contains all the modules required to execute the pipeline using any model provided in the Codes directory. 
The pipeline needs a Python version 3.8.6 and CUDA 11.7.0

## Testing the pipeline
Use the Final Pipeline.ipynb file to run the pipeline in Jupyter Notebook ipywidgets.

### User-editable configuration file
The integers inside the list are the stages of the pipeline the corresponding model can be used. Negative signs make the model the default choice for that stage.
zero (0) can be used to skip the model altogether. 

<img width="421" alt="gitlab2" src="https://github.com/tomrv22/content-depth-aware-restoration/assets/105001497/3ed51ed4-697a-4a04-af25-78c10ca57e5a">


### Execution of the pipeline
Users can use the "Automate" button to execute the first 4 stages of the pipeline after making their preferred choice of models for each stage. If unsatisfied with the execution of a step, the user can come back and resume the execution from that particular step using the separate "Process" button from each Tab. 

Before re-executing with another image, pipeline should be reset using the "Reset" button

<img width="496" alt="gitlab1" src="https://github.com/tomrv22/content-depth-aware-restoration/assets/105001497/124baf54-e2b6-44dd-8e66-4b4a8234935b">


#### Input image
The image to work on should be present inside the '/Thesis/Images' directory and should be selected from the main dropdown before the Tabs. 

![zebra_test2](https://github.com/tomrv22/content-depth-aware-restoration/assets/105001497/f7ff4819-4075-4a3c-9f85-63cba70237f1)

#### Custom layout and Fine-tuning

Fine Tuning stage allows the user to make prompt-based edits to individual objects or even background. The 'Image Editor' Tab permits layout customization using a simple dragging functionality. The link in this Tab takes the user to a web-based application running on another port of the machine. Here the user can reposition the objects in x, y, and z directions and even remove objects if they want to.

The "Default positions" button in the app allows one to maintain the original position of the objects from the parent image. Double-clicking any object in this editor sends the object to the background. Right-clicking allows the user to delete the objects. Clicking the "Save Canvas" button exits the app and saves the image which can be displayed in the "Restored Image" stage in the pipeline. 

The following video shows in detail each step that is explained above

https://github.com/tomrv22/content-depth-aware-restoration/assets/105001497/7ad8af98-3f56-4c27-b6ae-96d8638729ca

