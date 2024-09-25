# -*- coding: utf-8 -*-

"""
Skeleton for first part of the blob-detection coursework as part of INF250
at NMBU (Autumn 2017).
"""

__author__ = "Ivar Eftedal"
__email__ = "ivar.odegardstuen.eftedal@nmbu.com"

import matplotlib.pyplot as plt
import numpy as np
from skimage import io
from skimage.filters import threshold_otsu

filename = "gingerbreads.jpg"
fall = io.imread(filename)


def threshold(image, th=None):
    """Returns a binarised version of given image, thresholded at given value.

    Binarises the image using a global threshold `th`. Uses Otsu's method
    to find optimal thrshold value if the threshold variable is None. The
    returned image will be in the form of an 8-bit unsigned integer array
    with 255 as white and 0 as black.

    Parameters:
    -----------
    image : np.ndarray
        Image to binarise. If this image is a colour image then the last
        dimension will be the colour value (as RGB values).
    th : numeric
        Threshold value. Uses Otsu's method if this variable is None.

    Returns:
    --------
    binarised : np.ndarray(dtype=np.uint8)
        Image where all pixel values are either 0 or 255.
    """
    # Setup
    shape = np.shape(image)
    binarised = np.zeros([shape[0], shape[1]], dtype=np.uint8)

    if len(shape) == 3:
        image = image.mean(axis=2)
    elif len(shape) > 3:
        raise ValueError("Must be at 2D image")

    if th is None:
        th = threshold_otsu(image)

    # Start thresholding
    ## WRITE YOUR CODE HERE

    else:
        th = otsu(image)

    binarised[image > th] = 255
    binarised[image <= th] = 0

    return binarised


def histogram(image):
    """Returns the image histogram with 256 bins."""
    # Setup
    shape = np.shape(image)
    histogram = np.zeros(256)

    if len(shape) == 3:
        image = image.mean(axis=2)
    elif len(shape) > 3:
        raise ValueError("Must be at 2D image")

    # Start to make the histogram
    ## WRITE YOUR CODE HERE
    for i in range(shape[0]):
        for j in range(shape[1]):
            pixval = int(image[i, j])
            histogram[pixval] += 1

    return histogram


def otsu(image):
    """Finds the optimal thresholdvalue of given image using Otsu's method."""

    hist, bins = np.histogram(image, np.arange(0, 256))
    th = -1
    final_value = -1

    pixel_number = image.shape[0] * image.shape[1]
    mean_weight = 1.0 / pixel_number
    intensity_arr = np.arange(255)
    for t in bins[1:-1]:
        # bg = background, fg = foreground
        bg_probability = np.sum(hist[:t])
        fg_probability = np.sum(hist[t:])

        bg_weight = bg_probability * mean_weight
        fg_weight = fg_probability * mean_weight

        bg_mean_intensity = np.sum(intensity_arr[:t] * hist[:t]) / float(bg_probability)
        fg_mean_intensity = np.sum(intensity_arr[t:] * hist[t:]) / float(fg_probability)

        # Between class variance value:
        value = bg_weight * fg_weight * (bg_mean_intensity - fg_mean_intensity) ** 2

        if value > final_value:
            th = t
            final_value = value

    return th


# Display image
plt.imshow(fall)
plt.show()

# Histogram
histogram_plot = histogram(fall)
plt.plot(histogram_plot)
plt.show()

# Outsu value
outsu_value = otsu(fall)
print(f"Outsu value calculated: {outsu_value}")


# Display binarized image
binarized_image = threshold(fall, outsu_value)
plt.figure()
plt.imshow(binarized_image, cmap="gray")
plt.show()
