from matplotlib import pyplot as plt
import numpy as np
import kmeans
import sys
import cv2

def run_all():
    '''
    Driver function for default images and k values.
    '''
    image_names = ["chickens", "shibuya", "winter"]
    for name in image_names:
        loadpath = name + ".jpg"
        print("Processing image " + loadpath + "...")
        # Type of image is a numpy 
        image_orig = plt.imread(loadpath)
        # scale down the image so it's quicker for computing k-means
        # here we scale down the image to 50% of its original size using nearest neighbor interpolation
        # the resizing will not smooth the image, thus we should achieve the same results if running on
        # the original size image
        downsampled_image = cv2.resize(image_orig, (0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
        ks = [1,2,5,10,20]
        for k in ks:
            savepath = name + "-k" + str(k) + ".jpg"
            print("Initializing k=" + str(k) + " centers...")
            centers = kmeans.init_centers(k, downsampled_image)
            print("Running k means...")
            grid, centers = kmeans.run_kmeans(downsampled_image, centers)
            print("Writing image " + savepath + "...")
            image_edit = kmeans.create_new_image(grid, centers)
            # Upsample the image back to its original size to save it
            upsampled_image = cv2.resize(image_edit, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
            plt.imsave(savepath, upsampled_image)

def run_specific(filename, k):
    '''
    Driver function for an arbitrary filename and single k value.
    '''
    name, extension = filename.split(".")
    extension = "." + extension
    loadpath = name + extension
    print("Processing image " + loadpath + "...")
    # Type of image is a numpy nparray
    image_orig = plt.imread(loadpath)
    savepath = name + "-k" + str(k) + extension
    print("Initializing k=" + str(k) + " centers...")
    centers = kmeans.init_centers(k, image_orig)
    print("Running k means...")
    grid, centers = kmeans.run_kmeans(image_orig, centers)
    print("Writing image " + savepath + "...")
    image_edit = kmeans.create_new_image(grid, centers)
    plt.imsave(savepath, image_edit)

if (len(sys.argv) == 1):
    run_all()
elif (len(sys.argv) == 3):
    run_specific(sys.argv[1],int(sys.argv[2]))
else:
    print("Usage: \npython3 executable_name file_name k")
    sys.exit()
