#!/usr/bin/env python

# import the necessary packages
from re import M
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time

class SimpleKalmanFilter:
    def start(self, mea_e, est_e, q):
        self._err_measure = mea_e
        self._err_estimate = est_e
        self._q = q
        
    def updateEstimate(self, mea):
        self._kalman_gain = self._err_estimate/(self._err_estimate + self._err_measure)
        self._current_estimate = self._last_estimate + self._kalman_gain * (mea - self._last_estimate)
        self._err_estimate =  (1.0 - self._kalman_gain)*self._err_estimate + self.fabs(self._last_estimate-self._current_estimate)*_q
        self._last_estimate = self._current_estimate
        return self._current_estimate

    def setMeasurementError(self, mea_e):
        self._err_measure = mea_e
    
    def setEstimateError(self, est_e):
        self._err_estimate=est_e
        
    def etProcessNoise(self, q):
        self._q=q
    
    def getKalmanGain(self):
        return self._kalman_gain
        
    def getEstimateError(self):
        return self._err_estimate

class Nodo:
	def start(self):
		# Params
		self.kalman0 = SimpleKalmanFilter()
		self.kalman1 = SimpleKalmanFilter()
		self.kalman0.start(1, 1, 1)
		self.kalman1.start(1, 1, 1)
		self.image0 = None
		self.image1 = None
		self.br = CvBridge()
		self.tm = time.time()
        # Node cycle rate (in Hz).
		self.loop_rate = rospy.Rate(1)

        # Subscribers
		rospy.Subscriber("/camera0/usb_cam/image_raw",Image,self.callback0)
		rospy.Subscriber("/camera1/usb_cam/image_raw",Image,self.callback1)

	def callback0(self, msg):
		# rospy.loginfo('Image received...')
		self.image0 = self.br.imgmsg_to_cv2(msg)
		fr = self.findBall(self.image0)
		cv2.imshow("Camera 0", fr)
	
	def callback1(self, msg):
		# rospy.loginfo('Image received...')
		self.image1 = self.br.imgmsg_to_cv2(msg)
		fr = self.findBall(self.image1)
		cv2.imshow("Camera 1", fr)

	def findBall(self, img):
		# grab the current frame
		frame = img
		# if we are viewing a video and we did not grab a frame,
		# then we have reached the end of the video
		# resize the frame, blur it, and convert it to the HSV
		# color space
		frame = imutils.resize(frame, width=600)
		blurred = cv2.GaussianBlur(frame, (11, 11), 0)
		hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
		# construct a mask for the color "green", then perform
		# a series of dilations and erosions to remove any small
		# blobs left in the mask
		mask = cv2.inRange(hsv, self.greenLower, self.greenUpper)
		mask = cv2.erode(mask, None, iterations=2)
		mask = cv2.dilate(mask, None, iterations=2)

		# find contours in the mask and initialize the current
		# (x, y) center of the ball
		cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
			cv2.CHAIN_APPROX_SIMPLE)
		cnts = imutils.grab_contours(cnts)
		center = None
		# only proceed if at least one contour was found
		if len(cnts) > 0:
			# find the largest contour in the mask, then use
			# it to compute the minimum enclosing circle and
			# centroid
			c = max(cnts, key=cv2.contourArea)
			((x, y), radius) = cv2.minEnclosingCircle(c)
			M = cv2.moments(c)
			center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
			# only proceed if the radius meets a minimum size
			if radius > 10:
				# draw the circle and centroid on the frame,
				# then update the list of tracked points
				cv2.circle(frame, (int(x), int(y)), int(radius),
					(0, 255, 255), 2)
				cv2.circle(frame, center, 5, (0, 0, 255), -1)
		# update the points queue
		self.pts.appendleft(center)
		font                   = cv2.FONT_HERSHEY_SIMPLEX
		fontScale              = 1
		fontColor              = (255,0,0)
		thickness              = 1
		lineType               = 1

		bottomLeftCornerOfText = (40,60)
		cv2.putText(frame,'x: {}'.format(int(x)), 
			bottomLeftCornerOfText, 
			font, 
			fontScale,
			fontColor,
			thickness,
			lineType)
		
		bottomLeftCornerOfText = (40,100)
		cv2.putText(frame,'y: {}'.format(int(y)), 
			bottomLeftCornerOfText, 
			font, 
			fontScale,
			fontColor,
			thickness,
			lineType)

		bottomLeftCornerOfText = (40,140)
		cv2.putText(frame,'r: {}'.format(int(radius)), 
			bottomLeftCornerOfText, 
			font, 
			fontScale,
			fontColor,
			thickness,
			lineType)

		bottomLeftCornerOfText = (40,180)
		cv2.putText(frame,'T: {}s'.format(int(time.time() - self.tm)), 
			bottomLeftCornerOfText, 
			font, 
			fontScale,
			fontColor,
			thickness,
			lineType)
		# loop over the set of tracked points
		key = cv2.waitKey(1) & 0xFF
		return frame

		# cv2.imshow('camera0', img)

	def stereo(self):
		# define the lower and upper boundaries of the "green"
		# ball in the HSV color space, then initialize the
		# list of tracked points
		
		# self.greenLower = (19, 121, 85) # amarelo
		# self.greenUpper = (41, 255, 255)

		self.greenLower = (79, 75, 0)
		self.greenUpper = (91, 255, 255)
		self.pts = deque(maxlen=64)
		self.vs = self.image0
		# rospy.loginfo("aqui")

		# rospy.spin()
		# keep looping
		cont = 0
		while not rospy.is_shutdown():
			# img = self.image0
			rospy.loginfo(cont)
			cont = cont + 1
			# cv2.imshow('camera 0', self.image0)
			self.loop_rate.sleep()
		cv2.destroyAllWindows()

if __name__ == '__main__':
	rospy.init_node("StereoVisionCam", anonymous=True)
	my_node = Nodo()
	my_node.start()
	my_node.stereo()