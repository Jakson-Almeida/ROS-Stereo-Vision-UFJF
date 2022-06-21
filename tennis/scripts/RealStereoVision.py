#!/usr/bin/env python

# import the necessary packages
import rospy
from sensor_msgs.msg import Image
from imutils.video import VideoStream
from cv_bridge import CvBridge
import cv2
import imutils
import time

class Nodo:
	def start(self):
		# Params
		self.br = CvBridge()
        # Node cycle rate (in Hz).
		self.loop_rate = rospy.Rate(30)

        # Publishers
		self.pub0 = rospy.Publisher('/camera0/usb_cam/image_raw', Image, queue_size=10)
		self.pub1 = rospy.Publisher('/camera1/usb_cam/image_raw', Image, queue_size=10)

	def stereo(self):
		camOn = False
		v0 = VideoStream(0).start()
		v1 = VideoStream(2).start()
		time.sleep(2.0)
		rospy.loginfo("Publicando imagens webcam 0 e 1")
		while not rospy.is_shutdown():
			fr0 = v0.read()
			fr1 = v1.read()
			fr0 = imutils.resize(fr0, width=600)
			fr1 = imutils.resize(fr1, width=600)
			if fr0 is not None:
				image0 = self.br.cv2_to_imgmsg(fr0)
				if image0 is not None:
					self.pub0.publish(image0)
				#self.pub.publish(image0)
				if camOn:
					cv2.imshow("Camera 0", fr0)
			if fr1 is not None:
				image1 = self.br.cv2_to_imgmsg(fr1)
				if image1 is not None:
					self.pub1.publish(image1)
				#self.pub.publish(image1)
				if camOn:
					cv2.imshow("Camera 1", fr1)
			# rospy.loginfo(cont)
			# cv2.imshow('camera 0', self.image0)
			# self.loop_rate.sleep()
			key = cv2.waitKey(1) & 0xFF
			# if the 'q' key is pressed, stop the loop
			if key == ord("q"):
				break
		cv2.destroyAllWindows()

if __name__ == '__main__':
	rospy.init_node("RealStereoVision", anonymous=True)
	my_node = Nodo()
	my_node.start()
	my_node.stereo()