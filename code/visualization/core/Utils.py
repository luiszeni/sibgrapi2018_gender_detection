import cv2
import numpy as np
from core.BoundBox import BoundBox
from math import sqrt

def convert_detections_to_my_imp(out_boxes, out_scores, out_classes, class_names,shape):
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
        detections.append(BoundBox(left, top, right, bottom, classId=predicted_class, pred=score))
    return detections

def normalize_img(img, model_image_size):
    img = cv2.resize(img, model_image_size,  interpolation = cv2.INTER_CUBIC) 
    img = np.array(img, dtype='float32')
    img /= 255.
    img =  np.expand_dims(img, axis=0)
    return img

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)

    elif height is None:
        r = width / float(w)
        dim = (width, int(h * r))
    else:
        dim = (width, height)

    resized = cv2.resize(image, dim, interpolation = inter)

    return resized

#I really feel ashamed about this function =/
def the_worst_tracking(detections, detections_old, images):
    if detections_old is None:
        return images, detections
    ordedImgs = []
    ordedDetecs = []
    for det_old in detections_old:
        if len(images) == 0:
            break
        minDist = 10000
        img = images[0]
        indexMix = 0
        for i, det in enumerate(detections):
            dist = sqrt((det_old.xmin - det.xmin)**2 + (det_old.ymin - det.ymin)**2)
            if  minDist > dist:
                minDist = dist
                img = images[i]
                indexMix = i
        ordedImgs.append(img)
        ordedDetecs.append(detections[indexMix])
        del detections[indexMix]
        del images[indexMix]
    for img in images:
        ordedImgs.append(img)

    for det in detections:
        ordedDetecs.append(det)
    
    return ordedImgs,ordedDetecs


