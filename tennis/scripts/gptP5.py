from p5 import *

def setup():
    size(400, 400, P3D)
    no_loop()

def draw():
    camera = Camera()
    camera.attach()

    translate(width/2, height/2, 0)
    box(100)

run()