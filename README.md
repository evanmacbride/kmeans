# CISC684 Group 2 Homework 3

Group Members: Matthew Leinhauser, Evan MacBride, and Collin Meese

## kmeans

An implementation of the k-means algorithm for image segmentation. This program
will segment an image into a given number of k clusters centered on mean RGB
values.

### Images

Image input for step 1 (apple.jpg, cat.jpg, and leaf.jpg) and step 3
(headphones.jpg) can be found in `images/input`. Generated output images for
steps 1 and 3 are in `images/examples`. When running the program, new output
images will be saved in `images/output`.

### Running the program

To run the program on all three input images for step 1 for k values 1, 2, 5,
10, and 20, enter the following on the command line:

```python3 main.py
```

To run specific images at a given k value, enter

```python3 main.py <path_to_image>/<image_name> <K>
```

where K is an integer greater than zero. The path and image name for a good k=2
image for step 3 is `images/input/headphones.jpg`.

### Libraries

This project uses the functools lru_cache to reduce time spent calculating
distances between RGB vectors. Matplotlib is used for reading and writing image
files. NumPy is used for working with types and random numbers.
