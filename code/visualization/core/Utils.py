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
    return normalizeImg(img, model_image_size)