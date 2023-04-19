#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Pose
import csv

def callback(data):
    # Escreve os dados da mensagem em um arquivo CSV
    with open('position_data.csv', mode='a') as csv_file:
        fieldnames = ['x', 'y', 'z', 'orientation_x', 'orientation_y', 'orientation_z', 'orientation_w']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writerow({'x': data.position.x, 'y': data.position.y, 'z': data.position.z, 
                         'orientation_x': data.orientation.x, 'orientation_y': data.orientation.y,
                         'orientation_z': data.orientation.z, 'orientation_w': data.orientation.w})

if __name__ == '__main__':
    rospy.init_node('position_data_logger')
    rospy.Subscriber('/position', Pose, callback)
    rospy.spin()
