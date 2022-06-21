#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import time
import matplotlib.pyplot as plt
import rospy
from geometry_msgs.msg import Twist

x = np.linspace(0, 10, 100)
y = np.cos(x)

StereoPose = Twist()

def plot(x):
    for p in range(100):
        updated_y = np.cos(x-0.05*p)
        
        line1.set_xdata(x)
        line1.set_ydata(updated_y)
        
        figure.canvas.draw()
        
        figure.canvas.flush_events()
        time.sleep(0.05)

def  callback(msg):
    global StereoPose
    StereoPose = msg

if __name__ == '__main__':
    rospy.init_node("StereoPosePlot", anonymous=True)
    
    plt.ion()
    figure, ax = plt.subplots(figsize=(8,6))
    line1, = ax.plot(x, y)

    plt.title("Dynamic Plot of sinx",fontsize=25)

    plt.xlabel("X",fontsize=18)
    plt.ylabel("sinX",fontsize=18)

    loop_rate = rospy.Rate(10)
    rospy.Subscriber("stereo_pose",Twist,callback)
    # plot(x)

    while not rospy.is_shutdown():
        rospy.loginfo(StereoPose.linear.x)
        loop_rate.sleep()