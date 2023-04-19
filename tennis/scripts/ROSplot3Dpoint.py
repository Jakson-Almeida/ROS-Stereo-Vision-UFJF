#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import time
from pynput.keyboard import Key, Listener

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
key = 'a'

def get_ball_position():
    # retorna uma posição aleatória da bola
    return 100*np.random.rand(3)

positions = []

def on_press(k):
    print("algo")
    global key
    try:
        key = k.char
    except:
        key = k.name

def on_release(key):
    print("sei lá")
    if key == Key.esc:
        # Stop listener
        return False

with Listener(on_press=on_press,on_release=on_release) as listener:
        listener.join()

def record_positions():
    while key == ' ':
        position = get_ball_position()
        positions.append(position)

def plot_positions():
    print('plotando valores')
    xs, ys, zs = zip(*positions)
    ax.scatter(xs, ys, zs)

def plot():
    positions.clear()
    while key == ' ':
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
    print(key)
    if key == ' ':
        print("trem")
        plot()
        plot_positions()
        plt.show()
    time.sleep(0.01)

