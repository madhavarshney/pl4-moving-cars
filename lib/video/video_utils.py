import cv2
import numpy as np

def preprocess(img, shave=False, shavedim=(350,500, 500,1000)):
    #if the image is to be cropped make sure the values are sane first then reduce the image down to the new dimentions
    if shave:
        if(shavedim[0] < 0):
            shavedim[0] = 0
        if (shavedim[1] > img.shape[0]):
            shavedim[1] = img.shape[0]
        if (shavedim[2] < 0):
            shavedim[2] = 0
        if (shavedim[3] > img.shape[1]):
            shavedim[3] = img.shape[1]
        img = img[shavedim[0]:shavedim[1],shavedim[2]:shavedim[3]]
    sizexy = [img.shape[1], img.shape[0]]

    #get the appropriate padding on the image to make it square
    padhw = [0,0]
    if(sizexy[0] > sizexy[1]):
        dif = sizexy[0] - sizexy[1]
        border = cv2.copyMakeBorder(img, int(dif/2), int(dif/2), 0, 0, cv2.BORDER_CONSTANT, value=[200, 200, 200])
        padhw[0] = int(((dif/2)/border.shape[0]) * 448)

    elif (sizexy[1] > sizexy[0]):
        dif = sizexy[1] - sizexy[0]
        border = cv2.copyMakeBorder(img, 0, 0, int(dif / 2), int(dif / 2), cv2.BORDER_CONSTANT, value=[200, 200, 200])
        padhw[1] = int(((dif / 2) / border.shape[1]) * 448)
    else:
        border = img

    #resize the image to fit the 448,448 input that yolo requires
    border = cv2.resize(border, (448, 448))

    #yolo requires the image to be fed in by (channel, y,x). Transpose to match that.
    transposed = np.transpose(border, [2, 0, 1])
    return transposed, padhw, shavedim, border

def get_point(point):
    m_1= 0.6
    l_1= 140
    m_2= -0.35
    l_2= 390

    if point[1] >= m_2 * point[0] + l_2:
        return 0
    elif point[1] >= m_1 * point[0] + l_1:
        return 1
    else:
        return 2
