import matplotlib.pyplot as plt
import numpy as np

# reading image
filename = "gingerbreads.jpg"
from skimage import io

fall = io.imread(filename)

# display image
plt.imshow(fall)
plt.show()

# mean av 3 rgb bilder
imagemean = fall.mean(axis=2)

# histogram
shape = np.shape(imagemean)
K = 256
M = shape[0] * shape[1]
# M = shape[0]
histogram = np.zeros(256)
for i in range(shape[0]):
    for j in range(shape[1]):
        pixval = int(imagemean[i, j])
        histogram[pixval] += 1
plt.plot(histogram)
plt.show()
