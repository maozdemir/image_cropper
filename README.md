# image_cropper
A small Python script that uses OpenCV for detecting if the input images have faces that can fill at least 70% of a 512x512 frame.

Useful for creating custom LoRA's for Stable Diffusion.

Can be modified for detecting different objects.

Doesn't have the best accuracy, and face detection has been loosened in order to avoid ignoring false positives. The user is required to filter the results manually after the script is finished.

## Usage
Place your images in the `pics/` folder, and run the script.
