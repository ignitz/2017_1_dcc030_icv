#!/usr/bin/python
#pyobjviewer.py

"""
pyobjviewer is a simple viewer for obj format 3D meshes. It has no dependencies
on external 3D libraries such as opengl. It uses vanilla pygame, and provides
an example and insight into the inner workings of 3D display. lighting and
meshes.

This obj 3D file viewer is built on Orzel's amazingly simple
and facinating meshviewer script found at
http://labs.freehackers.org/wiki/pythonmeshviewer
The original meshviewer script has been slightly modified for the purposes of
this excercise, but you may obtain the original from the site listed above.

IMPORTANT NOTES
This script displays only triangle faces, so any meshes that have quad faces
will not display correctly. If possible triangulate the mesh first with a
3D application, such as wings3D 
Python is an interpreted language, and though this script will display low to
medium density meshes, it will be too slow for any pracital use other than
a display mechanism. It is meant purely for educational and fun purposes.
Please look at alternative mechanism for high speed mesh processing, such
as python opengl.

Usage:
python pygobjviewer.py modelname.obj

Jestermon
http://jestermon.weebly.com
jestermonster@gmail.com
"""

#dependencies
#pygame   http://www.pygame.rg

import os, sys, pygame, random, math, string
from pygame.locals import *
from MeshViewer import Point3D, Face, Triangle, Quad, Object

#Uses psyco for acceleration, though not required
try:
    import psyco
    psyco.full()
except:
    pass

#We use our own version of the env3d class
class Env3D:
    def __init__(self,screen,winsize):
    	self.winsize= winsize
    	self.zoom_factor = 1
    	self.light_vector_1 = Point3D (random.random(),random.random(),random.random())
    	self.light_vector_1.normalize()
    	self.light_vector_2 = Point3D (random.random(),random.random(),random.random())
    	self.light_vector_2.normalize()
    	self.screen = screen
    	self.wincenter = [winsize[0]/2, winsize[1]/2]
    	self.colorize = True

def loadObj(filename,o):
    try:
        fp = open(filename, "r")
    except:
        print "File: "+filename+" not found"
        sys.exit(1)
    for line in fp:
        if line.startswith('#'): continue
        values = line.split()
        if not values: continue
        if values[0] == 'v':
            v = map(float, values[1:4])
            o.points.append( Point3D(v[0],v[1],v[2]) )
        elif values[0] == 'f':
            p = []
            for v in values[1:]:
                w = v.split("/")
                p.append(w[0])
            #obj file uses 1 as base, adjust for indexing with -1
            o.faces.append( Triangle(int(p[0])-1,int(p[1])-1,int(p[2])-1))
    fp.close()
    return o   

#load the mesh
argv = sys.argv
if len(argv) < 2:
    meshname = "Pikachu.obj"
else:
    meshname = argv[1]
o = Object()
o.name = "Model"
o = loadObj(meshname,o)

#o.canonicalView()

#initialise pygame
size = width, height = 600, 400
screen = pygame.display.set_mode(size)
pygame.display.set_caption("PyObjViewer - A simple 3D viewer")
pygame.init()

#prepare 3d environment
random.seed()
env3d = Env3D(screen,[width,height])

#Setup display text
text = pygame.Surface((300, 200))
font = pygame.font.Font(None, 20)
white = (255,255,255)
fontimg = font.render("~~~~Keys~~~~",1,white)
text.blit(fontimg, (5,0))
fontimg = font.render("Left: Rotate left",1,white)
text.blit(fontimg, (5,20))
fontimg = font.render("Right: Rotate right",1,white)
text.blit(fontimg, (5,40))
fontimg = font.render("Up: Rotate up",1,white)
text.blit(fontimg, (5,60))
fontimg = font.render("Down: Rotate down",1,white)
text.blit(fontimg, (5,80))
fontimg = font.render("Page up: Scale up",1,white)
text.blit(fontimg, (5,100))
fontimg = font.render("Page down: Scale down",1,white)
text.blit(fontimg, (5,120))
fontimg = font.render("Space: Stop rotation",1,white)
text.blit(fontimg, (5,140))
fontimg = font.render("ESC: Quit",1,white)
text.blit(fontimg, (5,160))


def animate(activekey,o):
    if activekey == "L":
        o.rotateY(0.01)
    if activekey == "R":
        o.rotateY(-0.01)
    if activekey == "U":
        o.rotateX(0.01)
    if activekey == "D":
        o.rotateX(-0.01)


    return o

#Main loop
fps = 60
dt = 1.0/fps
clock = pygame.time.Clock()

activekey = ""

while 1:

    screen.fill((0,0,0))
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.display.quit()
            sys.exit(0)
        elif event.type == KEYDOWN:
            if event.key == K_ESCAPE:
                pygame.display.quit()
                sys.exit(0)
        try:
            if event.key == K_LEFT:
                activekey = "L"
            if event.key == K_RIGHT:
                activekey = "R"
            if event.key == K_UP:
                activekey = "U"
            if event.key == K_DOWN:
                activekey = "D"
            if event.key == K_SPACE:
                activekey = "STOP"
            if event.key == K_PAGEUP:
                o.scale(2.00)
            if event.key == K_PAGEDOWN:
                o.scale(0.50)
        except:
            pass
             
    o = animate(activekey,o)

    #First draw text, so that object appears in front of it
    screen.blit(text,(0,0))

    #display the 3D object
    o.display(env3d)

    pygame.display.update()
    clock.tick(fps)
