from shapedetection import ShapeDetector
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import imutils
import cv2

frame = cv2.imread("shapes.jpg")
cv2.imshow("Image", frame)
resized = imutils.resize(frame, width=300)
ratio = frame.shape[0] / float(resized.shape[0])
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray", gray)
blurred = cv2.GaussianBlur(gray, (7, 7), 0)
ret,thresh = cv2.threshold(blurred,230,255,cv2.THRESH_BINARY_INV)
cv2.imshow("Thresh", thresh)
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
sd = ShapeDetector()

for c in cnts:
	# compute the center of the contour, then detect the name of the
	# shape using only the contour
	M = cv2.moments(c)
	cX = int((M["m10"] / M["m00"]) * ratio)
	cY = int((M["m01"] / M["m00"]) * ratio)
	shape,color = sd.detect(c)
 
	# multiply the contour (x, y)-coordinates by the resize ratio,
	# then draw the contours and the name of the shape on the image
	c = c.astype("float")
	c *= ratio
	c = c.astype("int")
	cv2.drawContours(frame, [c], -1, (0, 255, 0), 2)
	cv2.putText(frame, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
		0.5, color, 2)
	# show the output image
	cv2.imshow("Image", frame)
	cv2.waitKey(0)
