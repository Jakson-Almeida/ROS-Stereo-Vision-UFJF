#!/usr/bin/env python
# -*- coding: utf-8 -*-

# roslaunch  openni2_launch  openni2.launch

# import the necessary packages
from re import M
import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose

from collections import deque
import argparse
import numpy as np
import math
import cv2
from cv_bridge import CvBridge
import imutils
import time

simulation = False

class StereoVision:
    def start(self):
        self.cam0 = Cam()
        self.cam1 = Cam()
        self.cam0.start()
        self.cam1.start()
        #self.initCamsThetaPhi(0.27, 0.25) #(0.54, 0.51)
        self.initCamsAngle(0.523599) 
        if(simulation):
            self.initCamsAngle(0.51)
            
    def initCamsAngle(self, angle):
		# Camera 0
        self.cam0.cx = -0.042
        self.cam0.cy = 0
        self.cam0.cz = 0
        if(simulation):
            self.cam0.cx = -0.5
            self.cam0.cz = 1
        self.cam0.setMaxAngle(angle)
		# Camera 1
        self.cam1.cx = 0.042
        self.cam1.cy = 0
        self.cam1.cz = 0
        if(simulation):
            self.cam1.cx = +0.5
            self.cam1.cz = 1
        self.cam1.setMaxAngle(angle)

    def initCamsThetaPhi(self, theta, phi):
        # Camera 0
        self.cam0.cx = -0.5
        self.cam0.cy = 0
        self.cam0.cz = 1
        self.cam0.setMaxTheta(theta)
        self.cam0.setMaxPhi(phi)
        # Camera 1
        self.cam1.cx = +0.5
        self.cam1.cy = 0
        self.cam1.cz = 1
        self.cam1.setMaxAngle(theta)
        self.cam1.setMaxPhi(phi)

    def update(self, tipo, px, py):
        if tipo == 0:
            cam = self.cam0
        else:
            cam = self.cam1
        cam.px = px
        cam.py = py
        cam.time = time.time()
        cam.calcTheta(px)
        cam.calcPhi(py)

    def stereoVision(self):
        # angles
        alpha0 = (math.pi / 2.0) - self.cam0.getTheta() if(self.cam0.getTheta() >= 0) else -self.cam0.getTheta() - (math.pi / 2.0)
        alpha1 = (math.pi / 2.0) - self.cam1.getTheta() if(self.cam1.getTheta() >= 0) else -self.cam1.getTheta() - (math.pi / 2.0)
        phi0   = self.cam0.getPhi()
        phi1   = self.cam1.getPhi()
        phi = (phi0+phi1) / 2.0

        # angular coeficients
        alpha0 = np.tan(alpha0)
        alpha1 = np.tan(alpha1)
        phi0 = np.tan(phi0)
        phi1 = np.tan(phi1)
        phi = np.tan(phi)

        if(alpha0 == alpha1): return
        
        # linear coeficiente
        b0 = self.cam0.cy - alpha0*self.cam0.cx
        b1 = self.cam1.cy - alpha1*self.cam1.cx

        self.cam0.Px = self.cam1.Px = (b1 - b0) / (alpha0 - alpha1)
        self.cam0.Py = self.cam1.Py = alpha0*self.cam0.Px + b0

        A = np.array([self.cam0.cx, self.cam0.cy])
        B = np.array([self.cam1.cx, self.cam1.cy])
        C = np.array([self.cam0.Px, self.cam0.Py])

        self.cam0.Pz = np.linalg.norm(A - C)*phi0 + self.cam0.cz
        self.cam1.Pz = np.linalg.norm(B - C)*phi0 + self.cam1.cz

        #print(self.cam0.Pz)

        C = np.array([self.cam0.Px, self.cam0.Py, self.cam0.Pz])
        self.cam0.rho = self.cam1.rho = np.linalg.norm(C)

        #print(np.linalg.norm(self.cam0.Pz))

        self.cam0.Pz = self.cam1.Pz = -self.cam0.rho*np.sin(self.cam0.getPhi()) + self.cam0.cz

class Cam:
	def start(self):
		self.cx = 0 # metro
		self.cy = 0 # metro
		self.cz = 0 # metro
		self.px = 0 # pixel
		self.py = 0 # pixel
		self.Px = 0 # metro
		self.Py = 0 # metro
		self.Pz = 0 # metro
		self.num_p_x = 600 #320
		self.num_p_y = 450 #240
		self.max_theta = 1 # radians
		self.theta = 1     # radians
		self.phi = 1       # radians
		self.rho = 1       # meters
		self.max_phi = 1   # radians
		self.max_d_x = 1   # meters
		self.max_d_y = 1   # meters
		self.time = 0

	def setMaxAngle(self, angle):
		self.max_theta = angle
		self.max_phi = angle
		self.max_d_x = np.abs(np.tan(angle))
		self.max_d_y = self.max_d_x

	def setMaxTheta(self, angle):
		self.max_theta = angle
		self.max_d_x = np.abs(np.tan(angle))

	def setMaxPhi(self, angle):
		self.max_phi = angle
		self.max_d_y = np.abs(np.tan(angle))

	def calcTheta(self, x):
		x = self.map(x, 0, self.num_p_x, -self.max_d_x, self.max_d_x)
		if np.abs(x) > self.max_d_x:
			self.theta = self.max_theta if (x > 0) else -self.max_theta
			return self.theta
		self.theta = np.arctan(x)
		return self.theta
	
	def calcPhi(self, y):
		y = self.map(y, 0, self.num_p_y, -self.max_d_y, self.max_d_y)
		if np.abs(y) > self.max_d_y:
			self.phi = self.max_phi if (y > 0) else -self.max_phi
			return self.phi
		self.phi = np.arctan(y)
		return self.phi

	def getTheta(self):
		return self.theta

	def getPhi(self):
		return self.phi

	def map(self, x, in_min, in_max, out_min, out_max):
		return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

class Nodo:
    def start(self):
        self.image0 = None
        self.image1 = None
        self.b_image0 = False
        self.b_image1 = False
        self.time_image0 = time.time()
        self.time_image1 = time.time()
        self.st = StereoVision()
        self.st.start()
        self.br = CvBridge()
        self.StereoPose = Pose()
        self.last_pose = (0, 0, 0)
        self.sub1 = None
        self.sub2 = None
        # Node cycle rate (in Hz).
        self.loop_rate = rospy.Rate(1)

        # Subscribers
        if(simulation):
            self.sub1 = rospy.Subscriber("/camera0/usb_cam/image_raw",Image,self.callback0)
            self.sub2 = rospy.Subscriber("/camera1/usb_cam/image_raw",Image,self.callback1)
        if not simulation:
            self.sub1 = rospy.Subscriber("/camera0/usb_cam/image_raw",Image,self.callback0)
            self.sub2 = rospy.Subscriber("/camera/rgb/image_raw",Image,self.callback1)
        self.pub = rospy.Publisher('/position', Pose, queue_size=10)

    def callback0(self, msg):
        self.b_image0 = True
        # rospy.loginfo('Image received...')
        self.image0 = self.br.imgmsg_to_cv2(msg)
        # print(time.time())
        # self.image0 = cv2.cvtColor(self.image0, cv2.COLOR_BGR2RGB)
    
    def callback1(self, msg):
        # self.time_image1 = time.time()
        self.b_image1 = True
        # rospy.loginfo('Image received...')
        self.image1 = self.br.imgmsg_to_cv2(msg)
        # self.image1 = cv2.cvtColor(self.image1, cv2.COLOR_BGR2RGB)
        # fr = self.findBall(1, self.image1)
        # cv2.imshow("Camera 1", fr)
    
    # def callback2(self, msg):
    #     # rospy.loginfo('Image received...')
    #     self.image0 = self.br.imgmsg_to_cv2(msg)
    #     # self.image0 = cv2.cvtColor(self.image0, cv2.COLOR_BGR2RGB)

    def findBall(self, tp, img, BGR=False):
        # grab the current frame
        frame = img
        if BGR:
            frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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
        self.mask0 = cv2.inRange(hsv, self.firstColorLower, self.firstColorUpper)
        self.mask0 = cv2.erode(self.mask0, None, iterations=2)
        self.mask0 = cv2.dilate(self.mask0, None, iterations=2)

        self.mask1 = cv2.inRange(hsv, self.secondColorLower, self.secondColorUpper)
        self.mask1 = cv2.erode(self.mask1, None, iterations=2)
        self.mask1 = cv2.dilate(self.mask1, None, iterations=2)

        mask = self.mask0 | self.mask1

        mask = cv2.dilate(mask, None, iterations=10)
        mask = cv2.erode(mask, None, iterations=10)

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
            self.st.update(tp, x, y)
            self.filter_pub(filter=False)
            self.pub.publish(self.StereoPose)
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
            cv2.putText(frame,'x: {:3.2f}'.format(self.st.cam0.Px), 
                bottomLeftCornerOfText, 
                font, 
                fontScale,
                fontColor,
                thickness,
                lineType)
        
            bottomLeftCornerOfText = (40,100)
            cv2.putText(frame,'y: {:3.2f}'.format(self.st.cam0.Py), 
                bottomLeftCornerOfText, 
                font, 
                fontScale,
                fontColor,
                thickness,
                lineType)

            bottomLeftCornerOfText = (40,140)
            cv2.putText(frame,'z: {:3.2f}'.format(self.st.cam0.Pz), 
                bottomLeftCornerOfText, 
                font, 
                fontScale,
                fontColor,
                thickness,
                lineType)
        # loop over the set of tracked points
        # key = cv2.waitKey(1) & 0xFF
        return frame
    
    def euclidian_dist(self, x0=0, x1=0, y0=0, y1=0, z0=0, z1=0):
        return math.sqrt((x1-x0)*(x1-x0) + (y1-y0)*(y1-y0) + (z1-z0)*(z1-z0))
    
    def filter_pub(self, filter=False, minDist=0, maxDist=0.6):
        self.st.stereoVision()

        if filter:
             x, y, z = (self.st.cam0.Px, self.st.cam0.Py, self.st.cam0.Pz)
             x0, y0, z0 = self.last_pose
             dist = self.euclidian_dist(x-x0, y-y0, z-z0)
             if dist < maxDist and dist >= minDist:
                self.StereoPose.position.x = self.st.cam0.Px
                self.StereoPose.position.y = self.st.cam0.Py
                self.StereoPose.position.z = self.st.cam0.Pz
                self.last_pose = (x, y, z)
        else:
            self.StereoPose.position.x = self.st.cam0.Px
            self.StereoPose.position.y = self.st.cam0.Py
            self.StereoPose.position.z = self.st.cam0.Pz
        
    def stereo(self):
        # define the lower and upper boundaries of the "green"
        # ball in the HSV color space, then initialize the
        # list of tracked points

        # colourfull ball
        # self.firstColorLower = (26, 149, 99)
        # self.firstColorUpper = (59, 255, 255)

        # self.secondColorLower = (0, 130, 145)
        # self.secondColorUpper = (16, 255, 255)

        # blue ball
        self.firstColorLower = (82, 110, 80)
        self.firstColorUpper = (153, 255, 255)

        self.secondColorLower = (82, 110, 80)
        self.secondColorUpper = (153, 255, 255)

        self.pts = deque(maxlen=64)
        self.vs = self.image0
        # rospy.loginfo("aqui")

        # rospy.spin()
        # keep looping
        cont = 0
        # key = cv2.waitKey(1) & 0xFF
        while not rospy.is_shutdown():
            # img = self.image0
            if self.image0 is not None and self.b_image0:
                # self.time_image0 = time.time()
                self.b_image0 = False
                fr = self.findBall(0, self.image0)
                cv2.imshow("Camera 0", fr)
                key = cv2.waitKey(1) & 0xFF
                # if key == ord("r") or key == ord("R"):
                #     mask = self.mask0 | self.mask1
                # print((time.time() - self.time_image0)*1000)
            if self.image1 is not None and self.b_image1:
                self.b_image1 = False
                if simulation:
                    fr = self.findBall(1, self.image1)
                else:
                    fr = self.findBall(1, self.image1, BGR=True)
                cv2.imshow("Camera 1", fr)
                # key = cv2.waitKey(1) & 0xFF
                # if key == ord("r") or key == ord("R"):
                #     mask = self.mask0 | self.mask1
                #     # cv2.imshow("Mascara 0", self.mask0)
                #     # cv2.imshow("Mascara 1", self.mask1)
                #     cv2.imshow("Mascara total 1", mask)

            # rospy.loginfo(cont)
            cont = cont + 1
            # cv2.imshow('camera 0', self.image0)
            # self.loop_rate.sleep()
        cv2.destroyAllWindows()
		

def get_arguments():
	ap = argparse.ArgumentParser()
	ap.add_argument('-s', '--simulation', required=False,
                    help='Use Gazebo simulation', action='store_true')
	return vars(ap.parse_args())

if __name__ == '__main__':
	rospy.init_node("StereoVisionCam", anonymous=True)
	# args = get_arguments()
	# if args['--simulation']:
	# 	simulation = True
	my_node = Nodo()
	my_node.start()
	my_node.stereo()