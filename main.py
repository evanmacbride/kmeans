from matplotlib import pyplot as plt
import numpy as np
import kmeans
import sys

def run_all():
    '''
    Driver function for default images and k values.
    '''
    image_names = ["apple", "cat", "leaf"]
    for name in image_names:
        loadpath = "images/input/" + name + ".jpg"
        print("Processing image " + loadpath + "...")
        image_orig = plt.imread(loadpath)
        ks = [1,2,5,10,20]
        for k in ks:
            savepath = "images/output/" + name + "-k" + str(k) + ".jpg"
            print("Initializing k=" + str(k) + " centers...")
            centers = kmeans.init_centers(k, image_orig)
            print("Running k means...")
            grid, centers = kmeans.run_kmeans(image_orig, centers)
            print("Writing image " + savepath + "...")
            image_edit = kmeans.create_new_image(grid, centers)
            plt.imsave(savepath, image_edit)

def run_specific(path, k):
    '''
    Driver function for an arbitrary filename and single k value.
    '''
    filename = path.split("/")[-1]
    name, extension = filename.split(".")
    extension = "." + extension
    print("Processing image " + path + "...")
    image_orig = plt.imread(path)
    savepath = "images/output/" + name + "-k" + str(k) + extension
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
    print("Usage: \npython3 main.py image_filename k")
    sys.exit()
