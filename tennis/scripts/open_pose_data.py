#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import rospy
from geometry_msgs.msg import Pose

def enviaDados():
    pub = rospy.Publisher('/position', Pose, queue_size=10)
    rospy.init_node('enviaDados', anonymous=True)
    rate = rospy.Rate(100) # 10hz

    with open('position_data.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                # Ignora o cabeçalho do arquivo CSV
                line_count += 1
            else:
                # Cria uma mensagem Pose a partir dos dados da linha
                pose_msg = Pose()
                pose_msg.position.x = float(row[0])
                pose_msg.position.y = float(row[1])
                pose_msg.position.z = float(row[2])
                pose_msg.orientation.x = float(row[3])
                pose_msg.orientation.y = float(row[4])
                pose_msg.orientation.z = float(row[5])
                pose_msg.orientation.w = float(row[6])

                # Publica a mensagem no tópico "/position"
                pub.publish(pose_msg)
                rospy.loginfo(pose_msg)

                line_count += 1
            rate.sleep()

if __name__ == '__main__':
    try:
        if not rospy.is_shutdown():
            enviaDados()
    except rospy.ROSInterruptException:
        pass
