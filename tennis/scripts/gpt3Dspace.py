import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import glfw
import random
import numpy as np

# import pyassimp

# scene = pyassimp.load('/home/jakson/catkin_ws/src/tennis/models/bola/BALLE DE TENNIS-rc.sldprt') #.sldprt

# Inicializa o GLFW
glfw.init()

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

# Loop principal
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
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

    # Desenhando a esfera
    glColor3f(1, 0, 0)

    glTranslatef(0, 0, 0)
    quad = gluNewQuadric()
    gluSphere(quad, 0.3, 30, 30)

    for i in range(50):
        x, y, z = ls_points[i]
        draw_ball(x, y, z)
    
    draw_trk(ls_points)

    glPopMatrix()

    # Atualizando a tela
    pygame.display.flip()

# Finalizando o Pygame e o PyOpenGL
pygame.quit()