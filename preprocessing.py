import numpy as np
import cv2
import matplotlib.pyplot as plt

x = np.loadtxt("./data/head_x.csv", delimiter=',')
x = x.reshape(-1, 64, 64).astype(np.uint8)

y = np.loadtxt("./data/head_y.csv", delimiter=',')
y = y.astype(np.uint8)

for t in range(1):
    image = x[t]

    for i in range(len(image)):
        for j in range(len(image[0])):
            image[i][j] = 255 if image[i][j] > 250 else 0


    print(y[t])

    _, contours, hierarchy = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contour_sizes = []
    largest_box = None
    largest_contour_area = 0.0
    
    for contour in contours:
        bounding_rectangle = cv2.minAreaRect(contour)
        box = cv2.boxPoints(bounding_rectangle)
        box = np.int0(box)
        area = cv2.contourArea(box)
        if area > largest_contour_area:
            largest_box = box
            largest_contour_area = area


    mask = np.zeros(image.shape, np.uint8)
    cv2.drawContours(mask, [largest_box], -1, 255, -1)
    cv2.bitwise_and(image, image, mask=mask)
    plt.imshow(mask, cmap='gray')
    plt.show()
    plt.imshow(image, cmap='gray')
    plt.show()


    

        












    # mask = np.zeros(image.shape, dtype=np.uint8)
    # cv2.drawContours(mask, [largest_box], -1, 255, -1)
    # # print(smallest_contour)
    # plt.imshow(mask, cmap='gray')
    # plt.show()
    # # plt.imshow(mask, cmap='gray')
    # plt.show()

