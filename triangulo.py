import sys
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from collections import namedtuple
import image
import random
import time

import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

Triangle = namedtuple("Triangle", "a,b,c")
Point = namedtuple("Point", "x,y,z")


def normalize(p):
    s = sum(u*u for u in p) ** 0.5
    return Point(*(u/s for u in p))


def midpoint(u, v):
    return Point(*((a+b)/2 for a, b in zip(u, v)))


def subdivide_hybrid3(tri, depth):
    def triangle(tri, depth):
        if depth == 0:
            yield tri
            return
        for t in subdivide_centroid(tri, 1):
            yield from edge(t, depth - 1)

    def centroid(tri, depth):
        if depth == 0:
            yield tri
            return
        for t in subdivide_midpoint(tri, 2):
            yield from triangle(t, depth - 1)

    def edge(tri, depth):
        if depth == 0:
            yield tri
            return
        for t in subdivide_edge(tri, 1):
            yield from centroid(t, depth - 1)

    return centroid(tri, depth)


def subdivide_hybrid2(tri, depth):
    def centroid(tri, depth):
        if depth == 0:
            yield tri
            return
        for t in subdivide_centroid(tri, 1):
            yield from edge(t, depth - 1)

    def edge(tri, depth):
        if depth == 0:
            yield tri
            return
        for t in subdivide_edge(tri, 1):
            yield from centroid(t, depth - 1)

    return centroid(tri, depth)


def subdivide_hybrid(tri, depth):
    def centroid(tri, depth):
        if depth == 0:
            yield tri
            return
        for t in subdivide_centroid(tri, 1):
            yield from edge(t, depth - 1)

    def edge(tri, depth):
        if depth == 0:
            yield tri
            return
        for t in subdivide_edge(tri, 1):
            yield from centroid(t, depth - 1)

    return edge(tri, depth)


def subdivide_midpoint2(tri, depth):
    if depth == 0:
        yield tri
        return
    #       p0
    #      /|\
    #     / | \
    #    /  |  \
    #   /___|___\
    # p1   m12   p2
    p0, p1, p2 = tri
    m12 = normalize(midpoint(p1, p2))
    # WRONG TRIANGULATION!
    yield from subdivide_midpoint2(Triangle(p0, m12, p1), depth-1)
    yield from subdivide_midpoint2(Triangle(p0, p2, m12), depth-1)


def subdivide_midpoint(tri, depth):
    if depth == 0:
        yield tri
        return
    #       p0
    #      /|\
    #     / | \
    #    /  |  \
    #   /___|___\
    # p1   m12   p2
    p0, p1, p2 = tri
    m12 = normalize(midpoint(p1, p2))
    yield from subdivide_midpoint(Triangle(m12, p0, p1), depth-1)
    yield from subdivide_midpoint(Triangle(m12, p2, p0), depth-1)


def subdivide_edge(tri, depth):
    if depth == 0:
        yield tri
        return
    #       p0
    #      /  \
    # m01 /....\ m02
    #    / \  / \
    #   /___\/___\
    # p1    m12   p2
    p0, p1, p2 = tri
    m01 = normalize(midpoint(p0, p1))
    m02 = normalize(midpoint(p0, p2))
    m12 = normalize(midpoint(p1, p2))
    triangles = [
        Triangle(p0,  m01, m02),
        Triangle(m01, p1,  m12),
        Triangle(m02, m12, p2),
        Triangle(m01, m02, m12),
    ]
    for t in triangles:
        yield from subdivide_edge(t, depth-1)


def subdivide_centroid(tri, depth):
    if depth == 0:
        yield tri
        return
    #       p0
    #       / \
    #      /   \
    #     /  c  \
    #    /_______\
    #  p1         p2
    p0, p1, p2 = tri
    centroid = normalize(Point(
        (p0.x + p1.x + p2.x) / 3,
        (p0.y + p1.y + p2.y) / 3,
        (p0.z + p1.z + p2.z) / 3,
    ))
    t1 = Triangle(p0, p1, centroid)
    t2 = Triangle(p2, centroid, p0)
    t3 = Triangle(centroid, p1, p2)

    yield from subdivide_centroid(t1, depth - 1)
    yield from subdivide_centroid(t2, depth - 1)
    yield from subdivide_centroid(t3, depth - 1)


def subdivide(faces, depth, method):
    for tri in faces:
        yield from method(tri, depth)
################

ancho = 720
alto = 576

rotaX = 0.0
rotaY = 0.0
rotaZ = 0.0
def drawFrame(ejex,ejey,ejez,X,Y,Z):
    drawOctaedro(ejex,ejey,ejez,X,Y,Z)
    pygame.display.flip()

def drawOctaedro(ejex,ejey,ejez,X,Y,Z):
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()

    glTranslatef(ejex, ejey, ejez)
    glRotatef(rotaX, 1.0, 0.0, 0.0)
    glRotatef(rotaY, 0.0, 1.0, 0.0)
    glRotatef(rotaZ, 0.0, 0.0, 1.0)

    glPushMatrix()
    glShadeModel(GL_SMOOTH)
    glBegin(GL_QUADS)


    colores=[[0.5, 0.5, 0.5],[0.3, 0.3, 0.3]]

    for i in range(0,len(X),3):
        color=colores[i%2]
        glColor3f(color[0],color[1],color[2])
        for j in range(3):
            glVertex3f(X[i+j], Y[i+j], Z[i+j])

    glEnd()
    glPopMatrix()

def resizeGL(ancho, alto):
    if alto == 0:
        alto = 1

    glViewport(0, 0, ancho, alto)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()

    gluPerspective(45, float(ancho)/float(alto), 0.1, 100.0)
    glMatrixMode(GL_MODELVIEW)


"""
Inicializa OpenGL
"""


def initGL():
    glClearColor(0.0, 0.0, 0.0, 0.0)
    glClearDepth(1.0)
    glDepthFunc(GL_LESS)

    glEnable(GL_DEPTH_TEST)
    glEnable(GL_POLYGON_SMOOTH)
    glEnable(GL_BLEND)

    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    glShadeModel(GL_SMOOTH)

    resizeGL(ancho, alto)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

################

if __name__ == '__main__':
    method = {
        "hybrid":   subdivide_hybrid,
        "hybrid2":  subdivide_hybrid2,
        "hybrid3":  subdivide_hybrid3,
        "midpoint": subdivide_midpoint,
        "midpoint2": subdivide_midpoint2,
        "centroid": subdivide_centroid,
        "edge":     subdivide_edge,
        }['edge']
    #cambiar al numero de triangulos 0 octaedro
    depth = int(sys.argv[1])
    color = getattr(cm, sys.argv[2] if len(sys.argv) >= 2 else 'coolwarm')

    # octahedron
    p = 2**0.5 / 2
    faces = [
        # top half
        Triangle(Point(0, 1, 0), Point(-p, 0, p), Point( p, 0, p)),
        Triangle(Point(0, 1, 0), Point( p, 0, p), Point( p, 0,-p)),
        Triangle(Point(0, 1, 0), Point( p, 0,-p), Point(-p, 0,-p)),
        Triangle(Point(0, 1, 0), Point(-p, 0,-p), Point(-p, 0, p)),

        # bottom half
        Triangle(Point(0,-1, 0), Point( p, 0, p), Point(-p, 0, p)),
        Triangle(Point(0,-1, 0), Point( p, 0,-p), Point( p, 0, p)),
        Triangle(Point(0,-1, 0), Point(-p, 0,-p), Point( p, 0,-p)),
        Triangle(Point(0,-1, 0), Point(-p, 0, p), Point(-p, 0,-p)),
    ]

    X = []
    Y = []
    Z = []
    T = []

    
    ##############################
    ejez = -8.0  # Alejamos, acercamos la camara
    ejex = 0  # Rotamos en X la vision
    ejey = 0  # Rotamos en X la vision

    glutInit(sys.argv)
    pygame.init()
    pygame.display.set_mode((ancho, alto), (OPENGL | DOUBLEBUF | GLUT_RGB))
    pygame.display.set_caption("Octaedro RGB")
    resizeGL(ancho, alto)
    initGL()
    for nivel in range(8):
        X = []
        Y = []
        Z = []
        for i, tri in enumerate(subdivide(faces, nivel, method)):
            X.extend([p.x for p in tri])
            Y.extend([p.y for p in tri])
            Z.extend([p.z for p in tri])
            #T.append([3*i, 3*i+1, 3*i+2])
        tiempo= time.time() 
        tiempof=tiempo+6   
        while tiempo<tiempof:
            #print(nivel)
            factorCambioY = 1.0
            factorCambioZ = -1.0
            factorCambioX = -1.0

            keys = pygame.key.get_pressed()

            #rotaX = rotaX + factorCambioX
            rotaY = rotaY + factorCambioY
            #rotaZ = rotaZ + factorCambioZ

            for eventos in pygame.event.get():
                if eventos.type == QUIT:
                    sys.exit(0)
                # if rotaY > 360 or rotaY == 0 or rotaY < -360:
                #    factorCambioY = factorCambioY * - 1.0
                
            drawFrame(ejex, ejey, ejez,X,Y,Z)
            tiempo= time.time() 

    #############################