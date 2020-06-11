import cv2, os
import numpy as np
import math, time

list_bird = []
list_xy = []

for i in range(500):
    for j in range(1, 500):

        alpha = math.atan(((i - 250) ** 2 + (j - 250) ** 2) ** 0.5 / 25)
        if i <= 250 and j > 250:
            beta = math.atan((250 - i) / (j - 250.0001))
        elif i < 250 and j <= 250:
            beta = math.atan((250 - i) / (j - 250.0001)) + math.pi
        elif i > 250 and j <= 250:
            beta = math.atan((i - 250) / (250.0001 - j)) + math.pi
        elif i >= 250 and j > 250:
            beta = 2 * math.pi - math.atan((i - 250) / (j - 250.0001))
        else:
            beta = math.pi

        x = (beta / (2 * math.pi)) * 3840
        if x >= 1:
            x -= 1

        y = 1920 - 960 * 2 / math.pi * alpha
        if y >= 1:
            y -= 1

        list_bird.append([i, j])
        list_xy.append([int(np.round(x)), int(np.round(y))])


# print(len(list_xy),len(list_bird))

def mapping(k, img1, img):
    # print(k,list_bird[k][0],list_bird[
    #
    #
    # cou = 0k][1],list_xy[k][0],list_xy[k][1])

    img1[list_bird[k][1]][list_bird[k][0]] = img[list_xy[k][1]][list_xy[k][0]]

    return img1
img_path = 'F:\\picture/'
pics = os.listdir(img_path)
pics = sorted(pics)

count = 0

for pic in pics:
    # count += 1
    if count % 1 == 0:
        count += 1
        img = cv2.imread(img_path + pic)
        img1 = np.zeros((500, 500, 3))
        t0 = time.clock()
        for k in range(len(list_xy)):
            img1 = mapping(k, img1, img)

        # cv2.imshow('ffff',img1/255.)
        # cv2.waitKey(10000)
        print(count)
        cv2.imwrite('F:\\test/' + pic, img1)
        print('time : ', 1 / (time.clock() - t0))
