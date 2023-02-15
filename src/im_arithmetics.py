from PIL import Image
from skimage.io import imsave
import numpy as np

class ImageArithmetics:
    def __inverse(n): 
        if n > 255: 
            n = 255
        if n < 0: n = 0
        return 255 - n
    

    def add(self, imp1, imp2):
        im1 = Image.open(imp1).convert("L")
        imp1_split = imp1.split(".")
        im1.save(f"{imp1_split[0]}_gray.{imp1_split[1]}")

        im2 = Image.open(imp2).convert("L")
        imp2_split = imp2.split(".")
        im2.save(f"{imp2_split[0]}_gray.{imp2_split[1]}")

        im1_arr = np.asarray(im1, dtype="uint16")
        im2_arr = np.asarray(im2, dtype="uint16")
        result = np.zeros(im1_arr.shape)

        for i in range(im1_arr.shape[0]):
            for j in range(im2_arr.shape[1]):
                sum = im1_arr[i, j] + im2_arr[i, j]
                if sum < 0: sum = 0
                result[i, j] = sum if sum <= 255 else 255

        imsave("addition.jpg", result)

    def subtract(self, imp1, imp2):
        im1 = Image.open(imp1).convert("L")
        imp1_split = imp1.split(".")
        im1.save(f"{imp1_split[0]}_gray.{imp1_split[1]}")

        im2 = Image.open(imp2).convert("L")
        imp2_split = imp2.split(".")
        im2.save(f"{imp2_split[0]}_gray.{imp2_split[1]}")
        
        im1_arr = np.asarray(im1, dtype="int16")
        im2_arr = np.asarray(im2, dtype="int16")
        result = np.zeros(im1_arr.shape)
        
        for i in range(im1_arr.shape[0]):
            for j in range(im2_arr.shape[1]):
                difference = im1_arr[i, j] - im2_arr[i, j]
                if difference >= 255: difference = 255
                result[i, j] = difference if difference >= 0 else 0
        
        imsave("substraction.jpg", result)

    def multiply(self, imp1, imp2):
        im1 = Image.open(imp1).convert("L")
        imp1_split = imp1.split(".")
        im1.save(f"{imp1_split[0]}_gray.{imp1_split[1]}")

        im2 = Image.open(imp2).convert("L")
        imp2_split = imp2.split(".")
        im2.save(f"{imp2_split[0]}_gray.{imp2_split[1]}")
        
        im1_arr = np.asarray(im1, dtype="uint16")
        im2_arr = np.asarray(im2, dtype="uint16")
        result = np.zeros(im1_arr.shape)
        
        for i in range(im1_arr.shape[0]):
            for j in range(im2_arr.shape[1]):
                product = im1_arr[i, j] * im2_arr[i, j]
                if product <= 0: product = 0
                result[i, j] = product if product <= 255 else 255
        
        imsave("multiplication.jpg", result)

    def divide(self, imp1, imp2):
        im1 = Image.open(imp1).convert("L")
        imp1_split = imp1.split(".")
        im1.save(f"{imp1_split[0]}_gray.{imp1_split[1]}")

        im2 = Image.open(imp2).convert("L")
        imp2_split = imp2.split(".")
        im2.save(f"{imp2_split[0]}_gray.{imp2_split[1]}")
        
        im1_arr = np.asarray(im1)
        im2_arr = np.asarray(im2)
        result = np.zeros(im1_arr.shape)
        
        for i in range(im1_arr.shape[0]):
            for j in range(im1_arr.shape[1]):
                quotient = im1_arr[i, j] / (im2_arr[i, j]+1)
                if quotient <= 0: quotient = 0
                result[i, j] = quotient if quotient <= 255 else 255
        
        print(result.min(), result.max())
        imsave("division.jpg", result)

    def inverse(self, imp1):
        im1 = Image.open(imp1).convert("L")
        imp1_split = imp1.split(".")
        im1.save(f"{imp1_split[0]}_gray.{imp1_split[1]}")

        im1_arr = np.asarray(im1)
        
        for i in range(im1_arr.shape[0]):
            for j in range(im1_arr.shape[1]):
                im1_arr[i, j] = self.__inverse(im1_arr[i, j])

        result_im = Image.fromarray(im1_arr)
        result_im.save("inverse.jpg")