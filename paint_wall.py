import argparse
import matplotlib.pyplot as plt
import cv2 
import numpy as np
import torch
import torchvision
import sys
import subprocess
from segment_anything import sam_model_registry, SamPredictor

""" 
@param {boolean} condition, condition of argument that will cause script to fail gracefully
@param {string} message, message to the user the cause of the failure
"""
def fail_message(condition: bool, message: str):
    if(condition):
        print(message)
        exit()

""" 
@param {array} mask, the pixel mask returned from the SAM model 
@param {string} ax, figure to plot on 
@return {array} the final image mask 
"""
def show_mask(mask: list, ax: plt, color: list) -> list:
    color = np.array([ int(color[0])/255, int(color[1])/255, int(color[2])/255, 1])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    return mask_image

""" 
@param {array} command, the commands you would like to excute in an array in the order you would like them to execute , 
"""
def execute_command(command: list):
    try:
        subprocess.check_call(command)
    except: 
        fail_message(True, "Failed to download the model, this is likely a problem on your end, you can try downloading manually. https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth")


## Parse args and check status
# https://docs.python.org/3/library/argparse.html
parser = argparse.ArgumentParser(description="Paint a segment", epilog='Example: python3 paint_wall.py filename=bedroom.jpg color=0,0,0,1 output=painted_bedroom.jpg')

## args
parser.add_argument('filename', help='path/to/image') 
parser.add_argument('color', help='format: 255,252,255') 
parser.add_argument('output', help='path/to/output/image/file') 
args = parser.parse_args()

image = args.filename.split('=')[-1]
color = args.color.split('=')[-1].split(',')
output = args.output.split('=')[-1]

fail_message(len(color) < 4 or len(color) > 4, 'invalid color, must be rgba format: 255,255,255, 1')
fail_message(color[3] and (float(color[3]) > 1 or float(color[3]) < 0), 'invalid alpha must be between 1 and 0')
fail_message(int(color[0]) < 0 or int(color[0]) > 255, 'Red color is out of bounds, must be between 0 and 255') 
fail_message(int(color[1]) < 0 or int(color[1]) > 255, 'Green color is out of bounds, must be between 0 and 255') 
fail_message(int(color[2]) < 0 or int(color[2]) > 255, 'Blue color is out of bounds, must be between 0 and 255') 

if torch.cuda.is_available() != True:
    print('Warning, torch is not using the gpu')

# Check to see if we have a model, else download it
try: 
    open('sam_vit_h_4b8939.pth')
except:
    print("SAM model is not found in the root dir, downloading...")
    execute_command([sys.executable, "-m", "pip", "install", "git+https://github.com/facebookresearch/segment-anything.git"])
    execute_command(['wget', " ",  'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth'])
    print("Finished downloading the model")

## setup the model from downloaded check point

sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device="cuda")

predictor = SamPredictor(sam)

## setup load image and get dimensions
try:
    image = cv2.imread(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, channels = image.shape 
except:
    fail_message(True, "Cannot open image file, please check the file in question exists and that it's an image file")

# add handler to open the image and window and allow the user to 
x_pix, y_pix = 0,0
def on_mouse_click(event, x, y, flags, param):
    global x_pix, y_pix
    if event == cv2.EVENT_LBUTTONDOWN:
        x_pix, y_pix = x, y
        print("You selected x:{} y: {}".format(x_pix, y_pix))

cv2.namedWindow("image")
cv2.setMouseCallback('image', on_mouse_click)
cv2.imshow('image',image)

print("Select a wall by clicking the left hand mouse button and close the image by typing the letter q")
while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Press 'q' to exit
        break

cv2.destroyAllWindows()

# Get the mask of the segment
predictor.set_image(image)
input_point = np.array([[x_pix, y_pix]])
input_label = np.array([1])

masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=False,
)

# write the image
masked = ''
for i, (mask, score) in enumerate(zip(masks, scores)):
    plt.figure(figsize=(height/100, width/100), dpi=100)
    plt.axis('off')
    plt.imshow(image)
    masked = show_mask(mask, plt.gca(), color)
    plt.savefig(output, bbox_inches='tight', pad_inches=0)
    plt.show()

