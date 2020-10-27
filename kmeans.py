from functools import lru_cache
import numpy as np
import sys

def init_centers(k, image):
    '''
    Initialize a centers list to k random RGB values taken from image.
    '''
    y = len(image)
    x = len(image[0])
    centers = []
    for i in range(k):
        ri = np.random.randint(0, y - 1)
        rj = np.random.randint(0, x - 1)
        centers.append(image[ri][rj])
    return centers

def run_kmeans(image, centers):
    threshold = 1
    prev_distance = 0
    cur_distance = 9999999
    while(abs(cur_distance - prev_distance) > threshold):
        prev_distance = cur_distance
        grid, cur_distance = update_grid(image, centers)
        centers = update_centers(grid, centers, image)
    return grid, centers

def update_grid(image, centers):
    '''
    Generate a new closest_grid that saves the index of the closest RGB center
    for each pixel in image. The dimensions of closest_grid mirror those of
    image. For tracking convergence, also return the distance_sum.
    '''
    closest_grid = np.full((len(image),len(image[0])), -1)
    distance_sum = 0
    for i in range(len(image)):
        for j in range(len(image[0])):
            closest_center, distance = assign_closest_center(image[i][j], centers)
            closest_grid[i][j] = closest_center
            distance_sum += distance
    return closest_grid, distance_sum

def update_centers(grid, centers, image):
    '''
    For each cell in each of the k clusters in grid, find the new center RGB
    value for that cluster by getting the average RGB value from the
    corresponding pixels in image.
    '''
    for k in range(len(centers)):
        rsum = 0
        bsum = 0
        gsum = 0
        cell_count = 0
        for y in range(len(grid)):
            for x in range(len(grid[0])):
                if (grid[y][x] == k):
                    rsum += image[y][x][0]
                    bsum += image[y][x][1]
                    gsum += image[y][x][2]
                    cell_count += 1
        ravg = int(rsum / cell_count)
        bavg = int(bsum / cell_count)
        gavg = int(gsum / cell_count)
        centers[k] = np.asarray([ravg, bavg, gavg], dtype=np.uint8)
    return centers

def assign_closest_center(pixel, centers):
    '''
    Get the index in the centers list of the closest center to pixel. For
    tracking convergence, also return the distance between that pixel and its
    closest center.
    '''
    least_distance = 999999
    closest_center = None
    it = (center for center in centers)
    for index, center in enumerate(it):
        dist = get_distance(pixel[0],pixel[1],pixel[2],center[0],center[1],center[2])
        if (dist < least_distance):
            least_distance = dist
            closest_center = index
    return closest_center, least_distance

@lru_cache(maxsize=None)
def get_distance(pr, pg, pb, cr, cg, cb):
    '''
    An lru_cache optimized distance function. Take the RGB values for two pixels
    and return the Euclidian distance between them.
    '''
    return np.sqrt((int(pr) - int(cr))**2 + (int(pg) - int(cg))**2 + (int(pb) - int(cb))**2)

def create_new_image(closest_grid, centers):
    '''
    Create an image matrix using the indices of centers we have saved in the
    closest_grid. centers contains RGB values for the centers of color clusters.
    '''
    distance_sum = 0
    new_image = []
    for i in range(len(closest_grid)):
        image_line = []
        for j in range(len(closest_grid[0])):
            index = closest_grid[i][j]
            pixel = centers[index]
            image_line.append(pixel)
        new_image.append(image_line)
    new_image = np.array(new_image, dtype=np.uint8)
    return new_image
