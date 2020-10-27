from matplotlib import pyplot as plt
import numpy as np
import kmeans

image_names = ["chickens", "shibuya", "winter"]

for name in image_names:
    loadpath = name + ".jpg"
    print("Processing image " + loadpath + "...")
    # Type of image is a numpy nparray
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
        plt.imsave(savepath, image_edit)
