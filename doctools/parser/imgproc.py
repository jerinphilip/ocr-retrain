import cv2
import numpy as np

def extract_units(image_location, units):
    image = cv2.imread(image_location)
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return_val, thresholded = cv2.threshold(grayscale, 0, 1, cv2.THRESH_OTSU)
    rows, cols = thresholded.shape

    def trimRect(unit):
        rectKeys = ['rectLeft', 'rectTop', 'rectRight', 'rectBottom']
        x, y, X, Y = list(map(lambda x: int(unit[x]), rectKeys))
        #print(x, y, X, Y)
        trim_v = lambda w: lambda v: max(0, min(v, w))
        x = trim_v(cols)(x)
        X = trim_v(cols)(X)
        y = trim_v(rows)(y)
        Y = trim_v(rows)(Y)
        return (x, X, y, Y)

    def extract_unit_image(unit):
        x, X, y, Y = trimRect(unit)
        subImg = thresholded[y:Y, x:X]
        # Truncate to height 32
        height, width = subImg.shape

        # Handle invalid cases.
        if height is 0 or width is 0:
            subImg = np.zeros((32, 32))

        height, width = subImg.shape
        ratio = 32/height
        resized = cv2.resize(subImg, None, fx=ratio, fy=ratio, 
                interpolation = cv2.INTER_CUBIC)
        return resized

    unitImages = list(map(extract_unit_image, units))
    return unitImages

