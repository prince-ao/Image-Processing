from PIL import Image
from skimage.io import imsave
import numpy as np
from skimage import exposure
import math
import cv2 as cv


class ImageEnhancement:

    def __stretch(self, pixel, pixel_max, pixel_min, min=0, max=255):
        return ((pixel-pixel_min)*((max-min)/(pixel_max-pixel_min)))+min

    def __gammaCorrection(self, pixel, C=1, a=1, GAMMA=1.9):
        return (C*(pixel + a))**GAMMA

    def __histogramEqualization(self, pixel, M, N, hist, L=256):
        pn = 0
        for i in range(pixel):
            pn += hist[pixel]
        return math.floor(((L-1)/M*N)*pn)
    
    def constrastStretch(self, imp):
        im = Image.open(imp).convert("L")
        im_arr = np.array(im, dtype="uint16")

        ma = np.amax(im_arr)
        mi = np.amin(im_arr)

        print(ma, mi)
        result = np.zeros(im_arr.shape)
        
        for i in range(im_arr.shape[0]):
            for j in range(im_arr.shape[1]):
                val = self.__stretch(im_arr[i, j], ma, mi)
                result[i, j] = val

        result.astype('uint8')
        imsave("contrast.jpg", result)
    
    def gammaCorrection(self, imp):
        im = Image.open(imp).convert("L")
        im_arr = np.asarray(im)

        result = np.zeros(im_arr.shape)

        for i in range(im_arr.shape[0]):
            for j in range(im_arr.shape[1]):
                result[i, j] = self.__gammaCorrection(im_arr[i, j], GAMMA=0.3)

        imsave("gamma.jpg", result)
    
    def histogramEqualization(self, imp):
        im = Image.open(imp).convert("L")
        im.save("gray.jpg")
        im_arr = np.asarray(im)

        new = exposure.equalize_hist(im_arr)

        imsave("histogram.jpg", new)

    def D(self, a, b):
        im1 = Image.open(a).convert("L")
        im2 = Image.open(b).convert("L")
        im1.save("gray1.jpg")
        im2.save("gray2.jpg")

        h1 = im1.histogram()
        h2 = im2.histogram()

        result = 0

        for val in range(len(h1)):
            result += abs(h1[val] - h2[val])

        return result

    def CLAHE(self, im):
        img = cv.imread(im,0)
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl = clahe.apply(img)
        cv.imwrite('clahe_2.jpg', cl)