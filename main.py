from matplotlib import pyplot as plt
import numpy as np
import kmeans

loadpath = "chickens-small.jpg"
savepath = "chickens-edit.jpg"
# Type of image is a numpy nparray
image_orig = plt.imread(loadpath)

k = 2
centers = kmeans.init_centers(k, image_orig)
grid, centers = kmeans.run_kmeans(image_orig, centers)
image_edit = kmeans.create_new_image(grid, centers)
plt.imsave(savepath, image_edit)
