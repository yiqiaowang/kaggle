import numpy as np
import cv2
import matplotlib.pyplot as plt


SIZE = 28



 
def rotate_and_slice(rectangle, image):
    rotation_matrix = cv2.getRotationMatrix2D(center=rectangle[0], angle=0, scale=1)

    new_size = image.shape[0]*3//2
    larger_image = np.zeros((new_size, new_size), dtype=np.uint8)
    rotation_matrix[0,2] += ((new_size // 2) - rectangle[0][0])
    rotation_matrix[1,2] += ((new_size // 2) - rectangle[0][1])

    image = cv2.warpAffine(src=image, M=rotation_matrix, dsize=(new_size,new_size))

    return image[new_size//2 - SIZE//2: new_size//2 + SIZE//2, new_size//2 - SIZE//2: new_size//2 + SIZE//2]



def preprocess(x):
    x = x.astype(np.uint8).reshape(-1, 64, 64)
    result = []
    for t in range(len(x)):
        image = x[t]


        for i in range(len(image)):
            for j in range(len(image[0])):
                image[i][j] = 255 if image[i][j] > 250 else 0



        _, contours, _ = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contour_sizes = []
        largest_box = None
        largest_rectangle = None
        largest_contour_area = 0.0

        for contour in contours:
            bounding_rectangle = cv2.minAreaRect(contour)
            square_area = max(bounding_rectangle[1])**2

            if square_area > largest_contour_area:
                box = cv2.boxPoints(bounding_rectangle)
                box = np.int0(box)
                largest_box = box
                largest_rectangle = bounding_rectangle
                largest_contour_area = square_area

        
        mask = np.zeros(image.shape, np.uint8)
        cv2.drawContours(mask, [largest_box], -1, 1, -1)
        for i in range(len(image)):
            for j in range(len(image[0])):
                image[i][j] = (image[i][j] & mask[i][j])



        
        image = rotate_and_slice(largest_rectangle, image)
        result.append(image)
        


    return np.array(result, dtype=np.uint8).reshape(-1, SIZE*SIZE)
    

    



