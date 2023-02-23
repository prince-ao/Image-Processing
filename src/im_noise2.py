import numpy as np
from skimage.io import imsave, imread
from skimage import color
from PIL import Image
import matplotlib.pyplot as plt
import math

class Utils:
    def save(self, im: np.ndarray, filename = "save.jpg"):
        """Save image"""
        imsave(filename, im)

    def save_rgb(self, r, g, b, filename_r = "red.jpg", filename_g = "green.jpg", filename_b = "blue.jpg"):
        """Save r, g, b channels seperetly"""
        self.save(self.make(r, "red"), filename_r)
        self.save(self.make(g, "green"), filename_g)
        self.save(self.make(b, "blue"), filename_b)
    
    def make(self, l: np.ndarray, color: str):
        """Make a red, green, or blue only image"""
        new = np.zeros((l.shape[0], l.shape[1], 3))
        if color == "red":
            new[:, :, 0] = l
            return new
        elif color == "green":
            new[:, :, 1] = l
            return new
        elif color == "blue":
            new[:, :, 2] = l
            return new
        else:
            raise ValueError("expected a valid color as input")
    
    def split_rgb(self, im: np.ndarray):
        """Split an rgb"""
        return im[:, :, 0], im[:, :, 1], im[:, :, 2]
    
    def combine_rgb(self, r: np.ndarray, g: np.ndarray, b: np.ndarray):
        """Combine an rgb"""
        new = np.zeros((r.shape[0], r.shape[1], 3))
        new[:, :, 0] = r
        new[:, :, 1] = g
        new[:, :, 2] = b
        return new

    def generate_histogram(self, im: np.ndarray):
        """Generate a histogram"""
        return im.flatten()
    
    def plot_histogram(self, h: np.ndarray):
        """Simple histogram plot"""
        plt.hist(h, bins=256)
        plt.show()
    
    def histogram_distance(self, h1: np.ndarray, h2: np.ndarray):
        """Returns histogram distance"""
        sum = 0
        for i in range(h1.shape[0]):
            sum += abs(h1[i] - h2[i])
        return sum

class ImageNoise:

    def __alpha_filter(self, im: np.ndarray, n = 9, m = 9, d = 10):
        result = np.zeros(im.shape)
        for i in range(im.shape[0]):
            for j in range(im.shape[1]):
                lis = []
                for k in range(n):
                    for l in range(m):
                        if (i-(n-2)+k) >= 0 and (i-(n-2)+k) < im.shape[0] and (j-(m-2)+l) >= 0 and (j-(m-2)+l) < im.shape[1]:
                            lis.append(im[(i-(n-2)+k), (j-(m-2)+l)])
                lis.sort()
                for i in range(math.floor(d/2)):
                    lis.pop(0)
                for i in range(math.floor(d/2)):
                    lis.pop()
                result[i, j] = np.array(lis).sum() / (n*m - d)
        return result

    def __median_filter(self, im: np.ndarray, n = 9, m = 9):
        result = np.zeros(im.shape)
        for i in range(im.shape[0]):
            for j in range(im.shape[1]):
                lis = []
                for k in range(n):
                    for l in range(m):
                        if (i-(n-2)+k) >= 0 and (i-(n-2)+k) < im.shape[0] and (j-(m-2)+l) >= 0 and (j-(m-2)+l) < im.shape[1]:
                            lis.append(im[(i-(n-2)+k), (j-(m-2)+l)])
                lis.sort()
                result[i, j] = lis[math.floor(len(lis)/2)]
        return result

    def __weighted_median_filter(self, im: np.ndarray, weight: np.ndarray):
        n, m = weight.shape[0], weight.shape[1]
        result = np.zeros(im.shape)
        for i in range(im.shape[0]):
            for j in range(im.shape[1]):
                lis = []
                for k in range(n):
                    for l in range(m):
                        if (i-(n-2)+k) >= 0 and (i-(n-2)+k) < im.shape[0] and (j-(m-2)+l) >= 0 and (j-(m-2)+l) < im.shape[1]:
                            for i in range(weight[k, l]):
                                lis.append(im[(i-(n-2)+k), (j-(m-2)+l)])
                lis.sort()
                result[i, j] = lis[math.floor(len(lis)/2)]
        return result
    
    def __multi_stage_filter(self, im: np.ndarray, n = 9, m = 9):
        result = np.zeros(im.shape)
        for i in range(im.shape[0]):
            for j in range(im.shape[1]):
                lis_r_d = []
                lis_l_d = []
                lis_v = []
                lis_h = []
                for k in range(n):
                    for l in range(m):
                        if (i-(n-2)+k) >= 0 and (i-(n-2)+k) < im.shape[0] and (j-(m-2)+l) >= 0 and (j-(m-2)+l) < im.shape[1]:
                            if(k == l):
                                lis_r_d.append(im[(i-(n-2)+k), (j-(m-2)+l)])
                            if k+l == n-1:
                                lis_l_d.append(im[(i-(n-2)+k), (j-(m-2)+l)])
                            if k == math.floor(n/2):
                                lis_v.append(im[(i-(n-2)+k), (j-(m-2)+l)])
                            if l == math.floor(m/2):
                                lis_h.append(im[(i-(n-2)+k), (j-(m-2)+l)])
                lis_r_d.sort()
                lis_l_d.sort()
                lis_v.sort()
                lis_h.sort()
                new_lis = [lis_r_d[math.floor(len(lis_r_d)/2)], lis_l_d[math.floor(len(lis_l_d)/2)], lis_v[math.floor(len(lis_v)/2)], lis_h[math.floor(len(lis_h)/2)]]
                result[i, j] = new_lis[math.floor(len(new_lis)/2)]
        return result

    def __contains(self, ls, z):
        for i in ls:
            if i == z: return True
        return False

    def __max_min_filter(self, im: np.ndarray, fn, n = 9, m = 9):
        result = np.zeros(im.shape)
        for i in range(im.shape[0]):
            for j in range(im.shape[1]):
                lis = []
                for k in range(n):
                    for l in range(m):
                        if (i-(n-2)+k) >= 0 and (i-(n-2)+k) < im.shape[0] and (j-(m-2)+l) >= 0 and (j-(m-2)+l) < im.shape[1]:
                            lis.append(im[(i-(n-2)+k), (j-(m-2)+l)])
                lis.sort()
                while(self.__contains(lis, 0)):
                    lis.remove(0)
                while(self.__contains(lis, 255)):
                    lis.remove(255)
                result[i, j] = lis[0 if fn == "min" else -1]
        return result

    def gen_gaussian(self, im: np.ndarray, mean = 10, varience = 0.8): # multicolor images
        util = Utils()
        noise = np.random.normal(mean, varience, size=(im.shape[0], im.shape[1]))

        r, g, b = util.split_rgb(im)
        util.save_rgb(r, g, b, "interior1_red.jpg", "interior_green.jpg", "interior_blue.jpg")

        r = r.astype(np.float64)
        g = g.astype(np.float64)
        b = b.astype(np.float64)
        r += noise
        g += noise
        b += noise
        util.save_rgb(r, g, b, "interior1_red_noise.jpg", "inerior_green_noise.jpg", "interior_blue_noise.jpg")

        result = util.combine_rgb(r, g, b)
        util.save(result, "interior1_noise.jpg")
        return result
    
    def gen_salt_and_pepper(self, im: np.ndarray, p_salt = 0.1, p_pepper = 0.1):
        util = Utils()

        r, g, b = util.split_rgb(im)
        util.save_rgb(r, g, b, "interior1_red.jpg", "interior1_green.jpg", "interior1_blue.jpg")

        mask = np.random.choice([0, 1, 2], size=r.shape, p=[p_salt, p_pepper, (1 - (p_salt + p_pepper))])
        for i in range(r.shape[0]):
            for j in range(r.shape[1]):
                if mask[i, j] == 0:
                    r[i, j], g[i, j], b[i, j] = 255, 255, 255
                elif mask[i, j] == 1:
                    r[i, j], g[i, j], b[i, j] = 0, 0, 0

        util.save_rgb(r, g, b, "interior1_red_noise.jpg", "interior1_green_noise.jpg", "interior1_blue_noise.jpg")
        result = util.combine_rgb(r, g, b)
        util.save(result, "interior1_noise.jpg")
        return result

    def median_filter(self, im: np.ndarray, n = 9, m = 9):
        util = Utils()
        
        r, g, b = util.split_rgb(im)
        util.save_rgb(r, g, b, "noise_red.jpg", "noise_green.jpg", "noise_blue.jpg")

        r_filtered, g_filtered, b_filtered = self.__median_filter(r, n, m), self.__median_filter(g, n, m), self.__median_filter(b, n, m)
        util.save_rgb(r_filtered, g_filtered, b_filtered, "noise_filtered_red.jpg", "noise_filtered_green.jpg", "noise_filtered_blue.jpg")

        result = util.combine_rgb(r_filtered, g_filtered, b_filtered)
        util.save(result, "noise_filtered.jpg")
        return result
    
    def min_filter(self, im: np.ndarray, n = 9, m = 9):
        util = Utils()
        
        r, g, b = util.split_rgb(im)
        util.save_rgb(r, g, b, "noise_red.jpg", "noise_green.jpg", "noise_blue.jpg")

        r_filtered, g_filtered, b_filtered = self.__max_min_filter(r, "min", n, m), self.__max_min_filter(g, "min", n, m), self.__max_min_filter(b, "min", n, m)
        util.save_rgb(r_filtered, g_filtered, b_filtered, "noise_filtered_red.jpg", "noise_filtered_green.jpg", "noise_filtered_blue.jpg")

        result = util.combine_rgb(r_filtered, g_filtered, b_filtered)
        util.save(result, "noise_filtered.jpg")
        return result

    def max_filter(self, im: np.ndarray, n = 9, m = 9):
        util = Utils()
        
        r, g, b = util.split_rgb(im)
        util.save_rgb(r, g, b, "noise_red.jpg", "noise_green.jpg", "noise_blue.jpg")

        r_filtered, g_filtered, b_filtered = self.__max_min_filter(r, "max", n, m), self.__max_min_filter(g, "max", n, m), self.__max_min_filter(b, "max", n, m)
        util.save_rgb(r_filtered, g_filtered, b_filtered, "noise_filtered_red.jpg", "noise_filtered_green.jpg", "noise_filtered_blue.jpg")

        result = util.combine_rgb(r_filtered, g_filtered, b_filtered)
        util.save(result, "noise_filtered.jpg")
        return result
    
    def alpha_trimmed_mean_filter(self, im: np.ndarray, n = 9, m = 9, d = 10):
        util = Utils()
        
        r, g, b = util.split_rgb(im)
        util.save_rgb(r, g, b, "noise_red.jpg", "noise_green.jpg", "noise_blue.jpg")

        r_filtered, g_filtered, b_filtered = self.__alpha_filter(r, n, m, d), self.__alpha_filter(g, n, m, d), self.__alpha_filter(b, n, m, d)
        util.save_rgb(r_filtered, g_filtered, b_filtered, "noise_filtered_red.jpg", "noise_filtered_green.jpg", "noise_filtered_blue.jpg")

        result = util.combine_rgb(r_filtered, g_filtered, b_filtered)
        util.save(result, "noise_filtered.jpg")
        return result

    def multi_stage_median_filter(self, im: np.ndarray, n = 9, m = 9):
        util = Utils()
        
        r, g, b = util.split_rgb(im)
        util.save_rgb(r, g, b, "noise_red.jpg", "noise_green.jpg", "noise_blue.jpg")

        r_filtered, g_filtered, b_filtered = self.__multi_stage_filter(r, n, m), self.__multi_stage_filter(g, n, m), self.__multi_stage_filter(b, n, m)
        util.save_rgb(r_filtered, g_filtered, b_filtered, "noise_filtered_red.jpg", "noise_filtered_green.jpg", "noise_filtered_blue.jpg")

        result = util.combine_rgb(r_filtered, g_filtered, b_filtered)
        util.save(result, "noise_filtered.jpg")
        return result

    def weighted_mean_filter(self, im: np.ndarray):
        weight = np.array([[1, 2, 1], [2, 3, 2], [1, 2, 1]])
        util = Utils()
        
        r, g, b = util.split_rgb(im)
        util.save_rgb(r, g, b, "noise_red.jpg", "noise_green.jpg", "noise_blue.jpg")

        r_filtered, g_filtered, b_filtered = self.__weighted_median_filter(r, weight), self.__weighted_median_filter(g, weight), self.__weighted_median_filter(b, weight)
        util.save_rgb(r_filtered, g_filtered, b_filtered, "noise_filtered_red.jpg", "noise_filtered_green.jpg", "noise_filtered_blue.jpg")

        result = util.combine_rgb(r_filtered, g_filtered, b_filtered)
        util.save(result, "noise_filtered.jpg")
        return result

    def sharpening(self, im: np.ndarray, n = 16, m = 16):
        util = Utils()
        
        r, g, b = util.split_rgb(im)
        util.save_rgb(r, g, b, "original_red.jpg", "original_green.jpg", "orignal_blue.jpg")

        r_filtered, g_filtered, b_filtered = self.__median_filter(r, n, m), self.__median_filter(g, n, m), self.__median_filter(b, n, m)
        util.save_rgb(r_filtered, g_filtered, b_filtered, "filtered_red.jpg", "filtered_green.jpg", "filtered_blue.jpg")

        result = util.combine_rgb(r_filtered, g_filtered, b_filtered)
        util.save(result, "filtered.jpg")

        grey = im - result
        util.save(grey, "grey.jpg")

        sharp = im + (0.2 * grey)
        util.save(sharp, "sharp.jpg")
        return sharp


        

imi = ImageNoise()
util = Utils()
im = imread("interior1_noise.jpg")
res = imi.median_filter(im)
#print(util.histogram_distance(util.generate_histogram(im), util.generate_histogram(im)))
