import cv2
import numpy as np

class BoundBox:
	def __init__(self, xmin, ymin, xmax, ymax, classId=None, pred=-1.0, imgID="", det=False, diff=False):
		self.xmin = float(xmin)
		self.ymin = float(ymin)
		self.xmax = float(xmax)
		self.ymax = float(ymax)
		self.pred = float(pred)
		self.imgID = imgID
		self.classId = classId
		self.det = det
		self.diff = diff
	def __repr__(self):
		return self.__str__()

	def __str__(self):

		if self.pred >= 0:
			return "(pred: " + str(self.pred) +  " xmin: " + str(self.xmin) + " ymin: " + str(self.ymin) + " xmax: " + str(self.xmax) + " ymax: " + str(self.ymax) + " classId: " + str(self.classId)+ ")"
		return "(xmin: " + str(self.xmin) + " ymin: " + str(self.ymin) + " xmax: " + str(self.xmax) + " ymax: " + str(self.ymax) + " classId: " + str(self.classId)+ ")"

	def drawInImage(self, img, color = (255,0,0), alpha = 1.0, scale = 1, lineWidth=1, drawpred=True, text=None):

		drawingImg = img.copy()

		cv2.rectangle(drawingImg,(int(self.xmin),int(self.ymin)),(int(self.xmax),int(self.ymax)),color,lineWidth)

		if self.pred >= 0 and drawpred:
			cv2.rectangle(drawingImg,(int(self.xmin),int(self.ymax)-20*scale),(int(self.xmin)+40*scale,int(self.ymax)),color,-1)
			cv2.putText(drawingImg,str(self.pred)[2:4] + '%',(int(self.xmin)+5,int(self.ymax)-5*scale), cv2.FONT_HERSHEY_SIMPLEX, 0.4*scale,(0,0,0),2,cv2.LINE_AA)
		
		if text != None:
			ymin = int(self.ymin)-20*scale

			if ymin < 0:
				ymin = 0

			cv2.rectangle(drawingImg,(int(self.xmin),ymin),(int(self.xmin)+ (len(text)*10*scale),ymin + 20*scale),color,-1)
			cv2.putText(drawingImg,text,(int(self.xmin)+5,ymin+15*scale), cv2.FONT_HERSHEY_SIMPLEX, 0.4*scale,(0,0,0),2,cv2.LINE_AA)

		cv2.addWeighted(drawingImg, alpha, img, 1 - alpha,	0, img)

	def iou(self, bb):
		# determine the coordinates of the intersection rectangle
		bi1 = max(self.xmin, bb.xmin)
		bi2 = max(self.ymin, bb.ymin)
		bi3 = min(self.xmax, bb.xmax)
		bi4 = min(self.ymax, bb.ymax)

		iw=bi3-bi1+1.
		ih=bi4-bi2+1.
		if iw>0 and ih>0:               
			# compute overlap as area of intersection / area of union
			ua=(bb.xmax-bb.xmin+1.)*(bb.ymax-bb.ymin+1.)+(self.xmax-self.xmin+1.)*(self.ymax-self.ymin+1.)-iw*ih;
			ov=iw*ih/ua;
			return ov
		return 0.0
