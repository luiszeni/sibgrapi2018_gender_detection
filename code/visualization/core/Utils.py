import cv2
import numpy as np
from core.BoundBox import BoundBox

def convertDetectionsToMyImp(out_boxes,out_scores,out_classes,class_names,shape):
    detections = []
    for i, c in enumerate(out_classes):
        predicted_class = class_names[c]
        box = out_boxes[i]
        score = out_scores[i]
        top, left, bottom, right = box
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(shape[1], np.floor(bottom + 0.5).astype('int32'))
        right = min(shape[2], np.floor(right + 0.5).astype('int32'))

        detections.append(BoundBox(left, top, right, bottom, 
            classId=predicted_class, pred=score))
    return detections

def normalizeImg(img, model_image_size):
    img = cv2.resize(img, model_image_size,  interpolation = cv2.INTER_CUBIC) 
    img = np.array(img, dtype='float32')
    img /= 255.
    img =  np.expand_dims(img, axis=0)
    return img

def loadAndNormalizeImg(inputImage, model_image_size):
    img = cv2.imread(inputImage)
    return normalizeImg(img, model_image_size), img.shape


def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    elif height is None:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))
    else:
        dim = (width, height)


    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized
