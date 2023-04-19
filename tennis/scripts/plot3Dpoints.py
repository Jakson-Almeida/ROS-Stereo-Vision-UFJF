#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import keyboard
import numpy as np
import time

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

def get_ball_position():
    # retorna uma posição aleatória da bola
    return 100*np.random.rand(3)

positions = []

def record_positions():
    while keyboard.is_pressed(' '):
        position = get_ball_position()
        positions.append(position)

def plot_positions():
    print('plotando valores')
    xs, ys, zs = zip(*positions)
    ax.scatter(xs, ys, zs)

def plot():
    positions.clear()
    while keyboard.is_pressed(' '):
        print('alguma coisa')
        position = get_ball_position()
        positions.append(position)
        time.sleep(0.01)

# Função que é chamada quando uma tecla é pressionada
# def on_press(event):
#     if event.key != 'space':
#         # Limpa o gráfico anterior
#         plt.close()

#         # Cria um novo gráfico
#         global fig, ax
#         fig = plt.figure()
#         ax = fig.add_subplot(111, projection='3d')
        
while True:
    if keyboard.is_pressed(' '):
        plot()
        plot_positions()
        plt.show()

