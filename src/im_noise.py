import math
import numpy as np
from PIL import Image
from skimage.io import imsave, imread
from skimage import color
import matplotlib.pyplot as plt

class ImageNoise:
    def arithmetic_filter(self, im: np.ndarray, n = 3, m = 3):
        result = np.zeros(im.shape)
        for i in range(im.shape[0]):
            for j in range(im.shape[1]):
                sum = 0
                for k in range(n):
                    for l in range(m):
                        if (i-(n-2)+k) >= 0 and (i-(n-2)+k) < im.shape[0] and (j-(m-2)+l) >= 0 and (j-(m-2)+l) < im.shape[1]:
                            sum += im[(i-(n-2)+k), (j-(m-2)+l)]
                result[i, j] = sum / (m*n)
        return result
    
    def weighted_average_filter(self, im: np.ndarray):
        n = 3
        m = 3
        filter_box = np.array([[1/16, 2/16, 1/16], [2/16, 4/16, 2/16], [1/16, 2/16, 1/16]])
        result = np.zeros(im.shape)
        for i in range(im.shape[0]):
            for j in range(im.shape[1]):
                sum = 0
                for k in range(n):
                    for l in range(m):
                        if (i-(n-2)+k) >= 0 and (i-(n-2)+k) < im.shape[0] and (j-(m-2)+l) >= 0 and (j-(m-2)+l) < im.shape[1]:
                            sum += filter_box[k, l] * im[(i-(n-2)+k), (j-(m-2)+l)]
                result[i, j] = sum
        return result

    def geometric_filter(self, im: np.ndarray, n = 3, m = 3):
        result = np.zeros(im.shape)
        im = im.astype("uint64")
        for i in range(im.shape[0]):
            for j in range(im.shape[1]):
                prod = 1
                for k in range(n):
                    for l in range(m):
                        if (i-(n-2)+k) >= 0 and (i-(n-2)+k) < im.shape[0] and (j-(m-2)+l) >= 0 and (j-(m-2)+l) < im.shape[1]:
                            prod *= im[(i-(n-2)+k), (j-(m-2)+l)]
                result[i, j] = prod ** (1/(n*m))
        return result
    
    def median_filter(self, im: np.ndarray, n = 3, m = 3):
        result = np.zeros(im.shape)
        for i in range(im.shape[0]):
            for j in range(im.shape[1]):
                lis = []
                for k in range(n):
                    for l in range(m):
                        if (i-(n-2)+k) >= 0 and (i-(n-2)+k) < im.shape[0] and (j-(m-2)+l) >= 0 and (j-(m-2)+l) < im.shape[1]:
                            lis.append(im[(i-(n-2)+k), (j-(m-2)+l)])
                    
                lis.sort()
                try:
                    lis.remove(255)
                    lis.remove(0)
                except ValueError:
                    pass
                result[i, j] = lis[math.floor(len(lis)/2)]
        return result
    
    def min_filter(self, im: np.ndarray, n = 3, m = 3):
        result = np.zeros(im.shape)
        for i in range(im.shape[0]):
            for j in range(im.shape[1]):
                lis = []
                for k in range(n):
                    for l in range(m):
                        if (i-(n-2)+k) >= 0 and (i-(n-2)+k) < im.shape[0] and (j-(m-2)+l) >= 0 and (j-(m-2)+l) < im.shape[1]:
                            lis.append(im[(i-(n-2)+k), (j-(m-2)+l)])
                    
                lis.sort()
                try:
                    lis.remove(255)
                except ValueError:
                    pass
                try:
                    lis.remove(0)
                except ValueError:
                    pass
                print(lis)
                result[i, j] = lis[0]
        return result
    
    def max_filter(self, im: np.ndarray, n = 3, m = 3):
        result = np.zeros(im.shape)
        for i in range(im.shape[0]):
            for j in range(im.shape[1]):
                lis = []
                for k in range(n):
                    for l in range(m):
                        if (i-(n-2)+k) >= 0 and (i-(n-2)+k) < im.shape[0] and (j-(m-2)+l) >= 0 and (j-(m-2)+l) < im.shape[1]:
                            lis.append(im[(i-(n-2)+k), (j-(m-2)+l)])
                    
                lis.sort()
                try:
                    while True:
                        lis.remove(255)
                except ValueError:
                    pass
                try:
                    while True:
                        lis.remove(0)
                except ValueError:
                    pass
                try:
                    result[i, j] = lis[-1]
                except IndexError:
                    result[i, j] = 125
        return result
                            
    def add_gausian_noise(self, im: np.ndarray):
        mean = im.mean()
        std = im.std()

        noise = np.random.normal(mean, std, size=im.shape)

        im+=noise

        im[im < 0] = 0
        im/=3

        return im

    def add_salt_and_peper_noise(self, im: np.ndarray):
        p = 0.05
        q = 0.05
        e = 0.90

        mask = np.random.choice([0, 1, 2], size=im.shape, p=[p, q, e])

        for i in range(im.shape[0]):
            for j in range(im.shape[1]):
                if mask[i, j] == 0:
                    im[i, j] = 0
                elif mask[i, j] == 1:
                    im[i, j] = 1

        return im
    
    def generate_histogram(self, im: np.ndarray, filename="histogram.jpg"):
        im *= 256
        fig, ax = plt.subplots()
        plt.hist(im.flatten(), bins=256)
        ax.set_title(f"mean = {im.mean()}, varience = {im.var()}")
        plt.show()
        

inn = ImageNoise()
#im = imread("gray.jpg")
im = imread("images/fashion_show4.jpg")
im = color.rgb2gray(im)
imsave("gray.jpg", im)
#result = inn.add_salt_and_peper_noise(im.copy())
#imsave("test.jpg", result)
filtered = inn.median_filter(im)
imsave("filtered.jpg", filtered)
#imsave("max_filter.jpg", filtered)
minus = im - filtered
imsave("grayminus.jpg", minus)
#minus= imread("grayminus.jpg")
a = 2
sharp = im + (0.1)*minus
imsave("sharp.jpg", sharp)