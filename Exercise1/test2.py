import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import threshold_otsu
from skimage import io

filename = "gingerbreads.jpg"
fall = io.imread(filename)

def threshold(image, th=None):
    """Returns a binarized version of given image, thresholded at given value."""
    # Setup
    shape = np.shape(image)
    binarised = np.zeros([shape[0], shape[1]], dtype=np.uint8)

    # If the image has 3 channels (RGB), convert it to grayscale by averaging
    if len(shape) == 3:
        image = image.mean(axis=2)  # Convert RGB to grayscale
    elif len(shape) > 3:
        raise ValueError('Must be a 2D image')

    # Use Otsu's method if no threshold is provided
    if th is None:
        th = otsu(image)

    # Apply thresholding
    binarised[image > th] = 255
    binarised[image <= th] = 0

    return binarised

def histogram(image):
    """Returns the image histogram with 256 bins."""
    shape = np.shape(image)
    histogram = np.zeros(256)

    if len(shape) == 3:
        image = image.mean(axis=2)  

    shape = np.shape(image)
    for i in range(shape[0]):
        for j in range(shape[1]):
            pixval = int(image[i, j])
            histogram[pixval] += 1

    return histogram

def otsu(image):
    """Finds the optimal threshold value of the given image using Otsu's method."""
    th = threshold_otsu(image)
    print(th)
    return th

# Display image
plt.imshow(fall)
plt.show()

# Display histogram
histogram_plot = histogram(fall)
plt.plot(histogram_plot)
plt.show()

# Display binarized image
binarized_image = threshold(fall)
plt.figure()
plt.imshow(binarized_image, cmap='gray')
plt.show()


