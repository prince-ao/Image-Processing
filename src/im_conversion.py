from PIL import Image
import numpy as np
from skimage.io import imsave, imread
from skimage import color

class ImageConversion:
    def __rgb_to_hsv(self, R, G, B):
        x_max = max(R, G, B)
        x_min = min(R, G, B)

        V = x_max
        C = x_max - x_min
        if x_max == x_min:
            return 0.0, 0.0, V
        S = (x_max - x_min) / x_max
        rc = (x_max-R) / C
        gc = (x_max-G) / C
        bc = (x_max-B) / C

        H = 0
        if R == V:
            H = bc-gc
        elif G == V:
            H = 2.0+rc-bc
        else:
            H = 4.0+gc-rc
        H = (H/6.0) % 1.0
        return H, S, V

    def __rgb_to_hsl(self, R, G, B):
        x_max = max(R, G, B)
        x_min = min(R, G, B)
        sumc = (x_max+x_min)
        rangec = (x_max-x_min)
        l = sumc/2.0
        if x_min == x_max:
            return 0.0, 1, 0.0
        if l <= 0.5:
            s = rangec / sumc
        else:
            s = rangec / (2.0-sumc)
        rc = (x_max-R) / rangec
        gc = (x_max-G) / rangec
        bc = (x_max-B) / rangec

        if R == x_max:
            h = bc-gc
        elif G == x_max:
            h = 2.0+rc-bc
        else:
            h = 4.0+gc-rc
        h = (h/6.0)%1.0
        return h, s, l
    
    def rbg_to_hsv(self, arr: np.ndarray, out_im_path="outputs/rbg_to_hsv.jpg"):
        result = np.zeros(arr.shape)
        if arr.max() > 1:
            arr /= 255
        r = arr[:, :, 0]
        g = arr[:, :, 1]
        b = arr[:, :, 2]

        for i in range(r.shape[0]):
            for j in range(r.shape[1]):
                result[i, j, 0], result[i, j, 1], result[i, j, 2] = self.__rgb_to_hsv(r[i, j], g[i, j], b[i, j])

        print(result.max(), result.min(), result.shape)
        return result

    def rgb_to_hsl(self, arr: np.ndarray):
        result = np.zeros(arr.shape)
        if arr.max() > 1:
            arr /= 255
        r = arr[:, :, 0]
        g = arr[:, :, 1]
        b = arr[:, :, 2]

        for i in range(r.shape[0]):
            for j in range(r.shape[1]):
                result[i, j, 0], result[i, j, 1], result[i, j, 2] = self.__rgb_to_hsl(r[i, j], g[i, j], b[i, j])

        print(result.max(), result.min(), result.shape)
        return result
    
    def rgb_to_grayscale_lightness(self, im: np.ndarray):
        result = np.zeros((im.shape[0], im.shape[1]))
        for i in range(im.shape[0]):
            for j in range(im.shape[1]):
                r = im[i, j, 0]
                g = im[i, j, 1]
                b = im[i, j, 2]
                result[i, j] = (max(r, g, b) + min(r, g, b))/2
        return result * 255
    
    def rgb_to_grayscale_average(self, im: np.ndarray):
        result = np.zeros((im.shape[0], im.shape[1]))

        for i in range(im.shape[0]):
            for j in range(im.shape[1]):
                r = im[i, j, 0]
                g = im[i, j, 1]
                b = im[i, j, 2]
                result[i, j] = (r + g + b)/3
        return result * 255
    
    def rgb_to_grayscale_luminosity(self, im: np.ndarray):
        result = np.zeros((im.shape[0], im.shape[1]))

        for i in range(im.shape[0]):
            for j in range(im.shape[1]):
                r = im[i, j, 0]
                g = im[i, j, 1]
                b = im[i, j, 2]
                result[i, j] = 0.21 * r + 0.72 * g + 0.07 * b
        return result * 255

    def rgb_to_grayscale_photopic(self, im: np.ndarray):
        result = np.zeros((im.shape[0], im.shape[1]))

        for i in range(im.shape[0]):
            for j in range(im.shape[1]):
                r = im[i, j, 0]
                g = im[i, j, 1]
                b = im[i, j, 2]
                result[i, j] = 0.256 * r + 0.67 * g + 0.065 * b
        return result * 255

    def rgb_to_grayscale_entropymine(self, im: np.ndarray):
        result = np.zeros((im.shape[0], im.shape[1]))

        for i in range(im.shape[0]):
            for j in range(im.shape[1]):
                r = im[i, j, 0]
                g = im[i, j, 1]
                b = im[i, j, 2]
                result[i, j] = (0.2126*r**2.2 + 0.7152*g**2.2 + 0.0722*b**2.2)**(1/2.2)
        return result * 255