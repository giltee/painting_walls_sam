# Paint a wall

## Intro
This project is to test out the ability of the SAM model by meta to predict wall segments and color to a desired color.

## Based off
https://github.com/facebookresearch/segment-anything/blob/main/notebooks/predictor_example.ipynb

## Dependencies 
- Pytorch
- OpenCV
- map

## Usage
```
python3 paint_wall.py -h // help
python3 paint_wall.py filename=bedroom.jpg color=0,0,0,1 output=painted_bedroom.jpg

# Wait for image to pop and select a point, press q to close the image
# The masked image will pop, you can close with q
# The image will be written to theh file in the output argument
```
## Outcome

The model does a good job at segmenting based on point location overall, with that being said there are small errors with segmentation on the majority of images I tested. Further work needs to be done either fine-tuning or using OpenCV to clean up the contours and handle shadows.