#!/usr/bin/env python
# -*- coding: utf-8 -*-

# importing libraries
import numpy as np
import time
import matplotlib.pyplot as plt
import cv2
import rospy
from geometry_msgs.msg import Twist

# creating initial data values
# of x and y
x = np.linspace(0, 10, 100)
y = np.sin(x)

# to run GUI event loop
plt.ion()

# here we are creating sub plots
figure, ax = plt.subplots(figsize=(10, 8))
line1, = ax.plot(x, y)

# setting title
plt.title("Posição da bola", fontsize=20)

# setting x-axis label and y-axis label
plt.xlabel("X-metro")
plt.ylabel("Y-metro")

# Loop

x_ros = []
y_ros = []
z_ros = []
p_ros = []

class Nodo:
    def start(self):
		# Params
        # Node cycle rate (in Hz).
        self.pose = Twist()
        self.loop_rate = rospy.Rate(60)

        # Subscribers
        self.sub = rospy.Subscriber('/stereo_pose', Twist, self.callback0)
    
    def callback0(self, msg):
		# rospy.loginfo('Image received...')
        self.pose = msg
        x_ros.append(msg.linear.x)
        y_ros.append(msg.linear.y)
        z_ros.append(msg.linear.z)
        p_ros.append(msg.linear.x, msg.linear.y, msg.linear.z)
    
    def plotagem(self):
        key = cv2.waitKey(1) & 0xFF
        while not rospy.is_shutdown():
            for _ in range(50):
                # creating new Y values
                new_y = np.sin(x-0.5*_)

                # updating data values
                line1.set_xdata(x)
                line1.set_ydata(new_y)

                # drawing updated values
                figure.canvas.draw()

                # This will run the GUI event
                # loop until all UI events
                # currently waiting have been processed
                figure.canvas.flush_events()
                # drawing updated values
                time.sleep(0.001)
            # self.loop_rate.sleep()
            if key == ord("r") or key == ord("R"):
                x_ros = []
                y_ros = []
                z_ros = []
        cv2.destroyAllWindows()



if __name__ == '__main__':
    rospy.init_node("grafico", anonymous=True)
    my_node = Nodo()
    my_node.start()
    my_node.plotagem()

# close all windows
cv2.destroyAllWindows()
