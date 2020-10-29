from matplotlib import pyplot as plt
import numpy as np
import kmeans
import sys
import cv2
import time

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
        ks = [1,2,5,10,20]
        for k in ks:
            savepath = name + "-k" + str(k) + ".jpg"
            print("Initializing k=" + str(k) + " centers...")
            centers = kmeans.init_centers(k, image_orig)
            print("Running k means...")
            grid, centers = kmeans.run_kmeans(image_orig, centers)
            print("Writing image " + savepath + "...")
            image_edit = kmeans.create_new_image(grid, centers)
            # Upsample the image back to its original size to save it
            plt.imsave(savepath, image_edit)

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
    print('Start:', time.time())
    run_all()
    print('Finish:', time.time())
elif (len(sys.argv) == 3):
    run_specific(sys.argv[1],int(sys.argv[2]))
else:
    print("Usage: \npython3 executable_name file_name k")
    sys.exit()
