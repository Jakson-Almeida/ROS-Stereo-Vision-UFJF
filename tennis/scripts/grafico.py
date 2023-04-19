#!/usr/bin/env python
# -*- coding: utf-8 -*-

# importing libraries
import numpy as np
import time
import matplotlib.pyplot as plt
import cv2
import rospy
from geometry_msgs.msg import Twist
import math
import random

# creating initial data values
# of x and y
# x = np.linspace(-10, 10, 100)
# y = np.sin(x)

# to run GUI event loop
# plt.ion()

# here we are creating sub plots
# figure, ax = plt.subplots(figsize=(10, 8))
# line1, = ax.plot(x, y)

# setting title
plt.title("Posição da bola", fontsize=20)

# setting x-axis label and y-axis label
plt.xlabel("X-metro")
plt.ylabel("Y-metro")

# Loop

time_start = 0
tempo_anterior = 0

x_ros = []
y_ros = []
z_ros = []
for i in range(50):
    x_ros.append(i)
    y_ros.append(i)
    z_ros.append(i)

class Nodo:
    def start(self):
        self.x0 = 0
        self.y0 = 0
        self.z0 = 0
        self.sizeList = 50
        for i in range(self.sizeList):
            x_ros.append(0)
            y_ros.append(0)
            z_ros.append(0)
		# Params
        # Node cycle rate (in Hz).
        self.pose = Twist()
        self.loop_rate = rospy.Rate(60)

        # Subscribers
        self.sub = rospy.Subscriber('/stereo_pose', Twist, self.callback0)
        time_start = time.time()

    def euclidian_dist(self, x0=0, x1=0, y0=0, y1=0, z0=0, z1=0):
        return math.sqrt((x1-x0)*(x1-x0) + (y1-y0)*(y1-y0) + (z1-z0)*(z1-z0))

    
    def callback0(self, msg):
		# rospy.loginfo('Image received...')
        self.pose = msg

        minDist = 0
        tempo = time.time() - time_start
        # 0.01
        x1 = msg.linear.x
        y1 = msg.linear.y
        z1 = msg.linear.z

        dist = self.euclidian_dist(self.x0, x1, self.y0, y1, self.z0, z1)

        self.x0 = msg.linear.x
        self.y0 = msg.linear.y
        self.z0 = msg.linear.z

        if dist >= minDist:
        # if tempo - tempo_anterior >= 0.05:
            print('tempo')
            x_ros.append(msg.linear.x)
            y_ros.append(msg.linear.y)
            z_ros.append(msg.linear.z)
            x_ros.pop(0)
            y_ros.pop(0)
            z_ros.pop(0)
        tempo_anterior = tempo
    
    def plotagem(self):
        key = cv2.waitKey(1) & 0xFF
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        
        while not rospy.is_shutdown():
            print(x_ros[1])

        # if key == ord("r") or key == ord("R"):
        for m, zlow, zhigh in [('o', -50, -25), ('^', -30, -5)]:
            xs = x_ros
            ys = y_ros
            zs = z_ros
            # for i in range(50):
            #     xs[i] += (random.random() - 0.5) * 3
            #     ys[i] += (random.random() - 0.5) * 3
            #     zs[i] += (random.random() - 0.5) * 3
            ax.scatter(xs, ys, zs, marker=m)
        
        plt.show()
        cv2.destroyAllWindows()



if __name__ == '__main__':
    rospy.init_node("grafico", anonymous=True)
    my_node = Nodo()
    my_node.start()
    my_node.plotagem()

# close all windows
cv2.destroyAllWindows()
