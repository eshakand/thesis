# Introduction 
This module introduces some of the utils that were utilized in our effort 

* contrast_enhancement: performs contrast enhancement on the image to emphasize the denser units of the mammogram. 
* convert_to_png: converts image from dcm folder to png
* densenet201: an experiment with a model other vgg16 
* determine_file: looks for malignant or benign files for classification
* enrich_Data: makes sur ethat all images are 5500 by 5500 and performs rotations. 
* find-size: returns a dictionary of all image sizes in a folder and subfolder 
* flood_filling_handler: performs flood filling to color in dense areas 
* image_segmentation: returns multiple images of specified size that are subimages of passed in image. 
* morphological closing: performs morphological closing to close in densre areas
* resize_handler: resize all images in a subdirectory to a specified size 
* test_classification: tests an individual image classification 
* test_ensemble: performs a voting ensemble method using multiple modules on a set of images and returns number of success 
* test_model: performs model evaluation on a subdirectory 
