import cv2
import numpy as np

def put_title(title, img, color = (255,255,255), background = (0,0,0), textScale=1.0, widthScale=1.0, normalize=True):
    top =  np.zeros((int(widthScale*40),img.shape[1],3))
    top[:,:,0] = background[0]
    top[:,:,1] = background[1]
    top[:,:,2] = background[2]

    cv2.putText(top,title,(10,int(widthScale*30)), cv2.FONT_HERSHEY_SIMPLEX, textScale, color, 2, cv2.LINE_AA)

    if normalize:
    	top = top/255

    ht= np.concatenate((top,img), axis=0)
    return ht