import pygame
from pygame.locals import *
from sys import exit
 
pygame.init()
screen=pygame.display.set_mode((560,420),0,16)
pygame.display.set_caption("Hello World")

dtheta = 3
frames = []
for k in range(0, 360, dtheta):
    filename = './Frames_560/Frame'+str(k)+'.png'
    frames.append(pygame.image.load(filename).convert())

for frame in frames:
    screen.blit(frame,(0,0))
    pygame.display.update()
    
exit()

