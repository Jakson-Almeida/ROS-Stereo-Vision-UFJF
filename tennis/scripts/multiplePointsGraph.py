#!/usr/bin/env python
# -*- coding: utf-8 -*-

# roslaunch  openni2_launch  openni2.launch

import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import glfw
import random
import numpy as np
import time
import matplotlib.pyplot as plt
import cv2
import rospy
from geometry_msgs.msg import Twist
import math
import numpy as np

# Inicializando o Pygame e o PyOpenGL
pygame.init()

width, height = 800, 600
screen = pygame.display.set_mode((width, height), DOUBLEBUF | OPENGL)

# Configurando a projeção 3D
gluPerspective(45, (width/height), 0.1, 50.0)
glTranslatef(0.0, 0.0, -5)

# Criando o objeto Rect do retângulo
rect = pygame.Rect(0, 0, 100, 100)

x_pos, y_pos = 0, 0

font = pygame.font.SysFont("Arial", 300, bold=True)

# Defina a mensagem a ser exibida
text = "Exemplo de banner"

ls_points = []

for i in range(50):
    raio = 3.0
    value = (raio*np.sin(i*np.pi/25.0), raio*np.cos(i*np.pi/25.0), 0)
    ls_points.append(value)

def draw_ball(x, y, z):
    glPushMatrix()
    glColor3f(1, 1, 0)
    glTranslatef(x, y, z)
    quad = gluNewQuadric()
    gluSphere(quad, 0.07, 30, 30)
    glPopMatrix()

def desenhar_linha_3d(x0, y0, z0, x1, y1, z1, espessura):
    glLineWidth(espessura)
    glBegin(GL_LINES)
    glVertex3f(x0, y0, z0)
    glVertex3f(x1, y1, z1)
    glEnd()

def draw_line3D(p1, p2):
    glColor3f(0, 1, 0)
    x0, y0, z0 = p1
    x1, y1, z1 = p2
    espessura = 5.0
    desenhar_linha_3d(x0, y0, z0, x1, y1, z1, espessura)

def draw_line2D(p1, p2):
    glPushMatrix()
    color = (0, 1, 0)
    thickness = 50
    pygame.draw.line(screen, color, p1, p2, thickness)
    glPopMatrix()

def draw_trk(ls, end=True):
    p1 = ls[0]
    p2 = ls[1]

    if end:
        p2 = ls[-1]
        draw_line3D(p1, p2)

    for i in range(len(ls)-1):
        p2 = ls[i+1]
        draw_line3D(p1, p2)
        # draw_line(p1, p2)
        p1 = p2

#################################################################################################

# setting title
plt.title("Posição da bola", fontsize=20)

# setting x-axis label and y-axis label
plt.xlabel("X-metro")
plt.ylabel("Y-metro")

# Loop

time_start = 0
tempo_anterior = 0
last_point = (0, 0, 0)

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
        # rospy.loginfo('Pose received...')
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
            # print('tempo')
            global last_point
            divis = 3.0
            last_point = msg.linear.x/divis, msg.linear.y/divis, msg.linear.z/divis
            x_ros.append(msg.linear.x)
            y_ros.append(msg.linear.y)
            z_ros.append(msg.linear.z)
            x_ros.pop(0)
            y_ros.pop(0)
            z_ros.pop(0)
        tempo_anterior = tempo
    
    def plotagem(self):
        global x_pos, y_pos
        key = cv2.waitKey(1) & 0xFF
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        
        while not rospy.is_shutdown():
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    break
                elif event.type == pygame.MOUSEMOTION:
                    x_pos, y_pos = event.pos
                    x_pos -= width/2
                    y_pos -= height/2
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Botão esquerdo do mouse pressionado
                        print("Botão esquerdo pressionado")
                    elif event.button == 3:  # Botão direito do mouse pressionado
                        print("Botão direito pressionado")
            # x_pos += random.random()*0.1

            # Limpando a tela
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            glPushMatrix()
            glRotatef(90, 1, 0, 0)
            glTranslatef(x_pos/160, y_pos/160, y_pos/160)

            glPushMatrix()
            glTranslatef(0, 0, 1.5)
            # glRotatef(90, 1, 0, 0)
            # Desenhando o retângulo na tela
            # pygame.draw.rect(screen, pygame.Color("red"), rect)

            glColor3f(0, 0.3, 0.3)
            glBegin(GL_QUADS)
            t1 = 3
            glVertex2f(-t1, -t1)
            glVertex2f(t1, -t1)
            glVertex2f(t1, t1)
            glVertex2f(-t1, t1)
            glEnd()
            glPopMatrix()

            glPushMatrix()
            # Desenhando a esfera
            glColor3f(1, 0, 0)
            # glTranslatef(0, 0, 0)
            x, y, z = last_point
            glTranslatef(x, y, z)
            quad = gluNewQuadric()
            gluSphere(quad, 0.3, 30, 30)
            glPopMatrix()

            draw_trk(ls_points)

            for i in range(50):
                x, y, z = ls_points[i]
                draw_ball(x, y, z)

            glPopMatrix()

            # Atualizando a tela
            pygame.display.flip()

        # Finalizando o Pygame e o PyOpenGL
        pygame.quit()



if __name__ == '__main__':
    rospy.init_node("grafico3D", anonymous=True)
    my_node = Nodo()
    my_node.start()
    my_node.plotagem()